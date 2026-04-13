import os
import json
import re
from typing import List, Optional, Literal

from pydantic import BaseModel, Field

from app.agents.deep_research_agent.model_registry import ModelRegistry
from app.agents.deep_research_agent.tools.tavily_search import search_prioritized, format_search_results
from app.tools.retrieval_tool import retrieval_tool


class EvidenceItem(BaseModel):
    id: str
    kind: Literal["kb", "web"]
    title: Optional[str] = None
    url: Optional[str] = None
    content: str


class EvidenceGrade(BaseModel):
    id: str
    relevant: bool = Field(description="Is this evidence relevant to the question?")
    relevance: float = Field(ge=0.0, le=1.0, description="Relevance score 0..1")
    credibility: float = Field(ge=0.0, le=1.0, description="Credibility score 0..1")
    notes: str


class EvidenceGradingResult(BaseModel):
    grades: List[EvidenceGrade]


def _retrieve_from_kb(query):
    try:
        result = retrieval_tool.invoke({"query": query})
        return result
    except Exception as e:
        print(f"KB retrieval error: {e}")
        return None


def _grade_evidence(question: str, items: List[EvidenceItem]) -> List[EvidenceGrade]:
    if not items:
        return []

    grader = ModelRegistry.get("grader")

    system_prompt = (
        "You are an evidence grader for a retrieval-augmented agriculture assistant. "
        "Given a farmer question and candidate evidence snippets, grade each item.\n\n"
        "Rules:\n"
        "- Relevance: Does this evidence directly help answer the question asked?\n"
        "- Credibility: Rate higher for official labels, extension/university, government; "
        "lower for SEO blogs, forums, pure marketing.\n"
        "- Do NOT be overly strict: if the question is about a branded product dosage/schedule, "
        "manufacturer label/brochure pages may be relevant even if commercial.\n"
        "- Output MUST be JSON via the provided schema." 
    )

    # Keep payload small to avoid context bloat
    payload = []
    for it in items:
        payload.append({
            "id": it.id,
            "kind": it.kind,
            "title": it.title,
            "url": it.url,
            "content": (it.content or "")[:1800],
        })

    user_prompt = (
        "Return ONLY valid JSON. No markdown. No extra text.\n"
        "Schema: {\"grades\": [{\"id\": string, \"relevant\": boolean, \"relevance\": number 0..1, \"credibility\": number 0..1, \"notes\": string}]}\n\n"
        f"Question:\n{question}\n\n"
        f"Evidence Items (JSON):\n{payload}"
    )

    try:
        raw = grader.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        text = (getattr(raw, "content", None) or str(raw) or "").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        parsed = json.loads(text)
        result = EvidenceGradingResult(**parsed)
        return list(result.grades or [])
    except Exception as e:
        print(f"Evidence grading error: {e}")
        # Fallback: keep everything but mark low credibility
        return [
            EvidenceGrade(id=it.id, relevant=True, relevance=0.5, credibility=0.3, notes="fallback")
            for it in items
        ]


def _extract_kb_text(kb_result):
    if isinstance(kb_result, str):
        return kb_result
    return kb_result.get("response_docs", "")


def _extract_kb_sources(kb_result):
    if isinstance(kb_result, str):
        return ["kb:local"]
    sources = []
    for doc in kb_result.get("reranked_top_docs", []):
        src = doc.get("source", "kb:local")
        if src not in sources:
            sources.append(src)
    return sources


WORKER_SUMMARY_PROMPT = """You are a research assistant for Indian agriculture.
Given the research data below, provide a factual summary answering the question.

STRICT RULES FOR NUMBERS AND DOSAGES:
- When extracting dosages, water volumes, timeframes, or any measurements from the
  research data, you MUST quote the EXACT numbers and units used in the source text.
- Do NOT convert units. Do NOT round numbers. Do NOT average numbers.
- If the text says "20g/ha", output exactly "20g/ha".
- If no exact number is listed for a measurement the question asks about, write
  "DATA_NOT_FOUND" for that specific metric instead of guessing.

SOURCE CREDIBILITY:
 - Use the provided credibility hints (if present) to decide how trustworthy a claim is.
 - Do not discard evidence purely because it is commercial if it contains an official label/brochure with exact dosage/schedule.
 - If a claim is from a low-credibility source, prepend it with "[LOW_CONFIDENCE_SOURCE]".

CITATION REQUIREMENT:
- For every factual claim (especially numbers, dosages, timings), append the source
  URL inline like this: [Source: https://example.com/page]
- This allows downstream systems to verify the information.

If the information comes from the Knowledge Base (KB) and no URL is available,
cite the KB source identifier shown in the research data, like:
[Source: KB:Agriculture-CPG-2020.pdf]

Keep the summary between 200 and 400 words.
Do not add information beyond what is in the research data."""


def worker_node(state: dict) -> dict:
    sub_question = state["sub_question"]
    strategy = state["strategy"]
    localization = state.get("localization", "India")
    dynamic_instruction = state.get("worker_instruction", "Extract relevant facts and figures.")
    # Use planner-generated academic search keywords if available
    search_keywords = state.get("search_keywords", sub_question)

    print(f"Worker: {sub_question} [{strategy}]")
    print(f"Instruction: {dynamic_instruction}")
    print(f"Search Keywords: {search_keywords}")

    kb_text = ""
    kb_sources = []
    web_text = ""
    web_sources = []
    web_results = []

    if strategy in ("kb_only", "kb_then_web"):
        localized_query = f"{sub_question} {localization}"
        kb_result = _retrieve_from_kb(localized_query)
        if kb_result:
            kb_text = _extract_kb_text(kb_result)
            kb_sources = _extract_kb_sources(kb_result)
            if kb_text:
                print(f"Worker: KB returned {len(kb_text)} chars")

    if strategy == "web_only" or strategy == "kb_then_web":
        # Use the planner's academic/extension search keywords for better results
        search_query = f"{search_keywords} {localization} India"
        print(f"Worker: searching web for '{search_query}'")
        max_results = int(os.getenv("TAVILY_MAX_RESULTS", "10"))
        web_results = search_prioritized(search_query, max_results=max_results)
        web_text = format_search_results(web_results)
        web_sources = [r["url"] for r in web_results if r.get("url")]
        print(f"Worker: web returned {len(web_results)} results")

    # LLM-based relevance grading (KB + each web result). This replaces rule-based sufficiency.
    evidence_items: List[EvidenceItem] = []
    if kb_text:
        evidence_items.append(EvidenceItem(id="kb", kind="kb", content=kb_text[:3500]))
    for i, r in enumerate(web_results or []):
        evidence_items.append(EvidenceItem(
            id=f"web_{i}",
            kind="web",
            title=r.get("title"),
            url=r.get("url"),
            content=(r.get("content") or "")[:2500],
        ))

    grades = _grade_evidence(sub_question, evidence_items)
    grade_by_id = {g.id: g for g in grades}

    # Apply grading decisions
    kb_grade = grade_by_id.get("kb")
    if kb_grade and (not kb_grade.relevant or kb_grade.relevance < 0.35):
        kb_text = ""
        kb_sources = []
        print(f"Worker: KB dropped by grader")

    kept_web_results = []
    for i, r in enumerate(web_results or []):
        g = grade_by_id.get(f"web_{i}")
        if not g:
            continue
        if g.relevant and g.relevance >= 0.35:
            r2 = dict(r)
            r2["credibility"] = g.credibility
            r2["relevance"] = g.relevance
            kept_web_results.append(r2)
    if web_results and not kept_web_results:
        print("Worker: web results dropped by grader (no relevant items)")

    web_results = kept_web_results
    web_text = format_search_results(web_results)
    web_sources = [r["url"] for r in web_results if r.get("url")]

    combined_data = ""
    if kb_text:
        kb_src_line = ""
        if kb_sources:
            kb_src_line = "KB Sources: " + ", ".join(kb_sources) + "\n"
        combined_data += f"Knowledge Base Results:\n{kb_src_line}{kb_text}\n\n"
    if web_text:
        combined_data += f"Web Search Results:\n{web_text}\n\n"

    if not combined_data.strip():
        print("Worker: no data found")
        return {
            "worker_reports": [{
                "sub_question": sub_question,
                "answer_found": False,
                "findings": "No relevant information found for this question.",
                "confidence": "low",
                "sources": [],
                "data_points": [],
                "contradictions": None,
            }]
        }

    llm = ModelRegistry.get("worker")
    
    user_prompt = (
        f"Question: {sub_question}\n"
        f"Specific Instructions: {dynamic_instruction}\n\n"
        f"Research Data:\n{combined_data}"
    )

    try:
        response = llm.invoke([
            {"role": "system", "content": WORKER_SUMMARY_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        findings = response.content.strip()
    except Exception as e:
        print(f"Worker summarization error: {e}")
        findings = combined_data[:1500]

    all_sources = kb_sources + web_sources
    confidence = "high" if kb_text and web_text else ("medium" if kb_text or web_text else "low")

    print(f"Worker: done | confidence={confidence} | sources={len(all_sources)}")

    return {
        "worker_reports": [{
            "sub_question": sub_question,
            "answer_found": True,
            "findings": findings,
            "confidence": confidence,
            "sources": all_sources,
            "data_points": [],
            "contradictions": None,
        }]
    }


