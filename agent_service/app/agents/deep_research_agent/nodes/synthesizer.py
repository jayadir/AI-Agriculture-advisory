import json
from typing import Any
from app.agents.deep_research_agent.model_registry import ModelRegistry
from app.agents.deep_research_agent.models import SynthesisResult
from app.agents.deep_research_agent.state import DeepResearchState


def _extract_first_json_object(text: str) -> str | None:
    """Extract the first top-level JSON object from a string.

    Uses a small state machine to find a balanced {...} region while respecting
    quoted strings and escape sequences.
    """
    if not text:
        return None

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def _json_loads_relaxed(text: str) -> dict:
    """Parse JSON with a couple of best-effort fallbacks.

    LLMs sometimes emit literal newlines or other control characters inside JSON
    strings. `strict=False` accepts these.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Allow control characters inside strings.
        return json.loads(text, strict=False)


def _coerce_str_list(value: Any) -> list[str]:
    """Coerce a value into a list[str].

    LLM JSON often returns lists of objects like {"url": ...} even when the schema
    asks for strings. We normalize to strings and preserve order.
    """
    if value is None:
        return []

    items: list[Any]
    if isinstance(value, list):
        items = value
    else:
        items = [value]

    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        s = ""
        if item is None:
            s = ""
        elif isinstance(item, str):
            s = item
        elif isinstance(item, dict):
            # Common shapes: {"url": "..."}, {"source": "..."}, {"link": "..."}
            s = (
                item.get("url")
                or item.get("source")
                or item.get("link")
                or item.get("href")
                or ""
            )
            if not s:
                # Last resort: stable string representation
                s = json.dumps(item, ensure_ascii=False)
        else:
            s = str(item)

        s = (s or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)

    return out


def _retry_from_worker_reports(worker_reports: list[dict]) -> list[str]:
    """Pick sub-questions that likely need a retry based on worker outputs."""
    retry_qs: list[str] = []
    for r in worker_reports or []:
        sq = (r.get("sub_question") or "").strip()
        findings = (r.get("findings") or "")
        answer_found = bool(r.get("answer_found"))
        conf = (r.get("confidence") or "").lower().strip()

        missing = ("DATA_NOT_FOUND" in findings) or ("No relevant information found" in findings)
        low_conf = conf in {"low"}

        if sq and ((not answer_found) or missing or low_conf):
            retry_qs.append(sq)

    # De-dupe while preserving order
    deduped: list[str] = []
    seen: set[str] = set()
    for q in retry_qs:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    return deduped


SYNTHESIZER_SYSTEM_PROMPT = """You are a research synthesis expert for Indian agriculture advisory.

Given multiple worker research reports, you must:
1. Merge findings from all reports into a single coherent answer.
2. If numbers or facts conflict between reports, prefer data that appears more credible
    (official labels / extension / government), and note the contradiction briefly.
3. If any report has low confidence or no data, decide if re-research would help.
4. Set needs_retry to true ONLY if critical information is completely missing AND
   you believe a differently worded search would find it. Do not retry for minor gaps.

CITATION AND VERIFICATION RULES:
- PRESERVE all inline [Source: URL] citations from worker reports in your synthesized answer.
- If a dosage or number has NO citation, mark it as [NO_SOURCE] in your synthesized answer.
- Do NOT invent any numbers, dosages, or timings that are not present in the worker reports.
- If a specific metric was reported as DATA_NOT_FOUND by workers, preserve that status.

Your synthesized_answer should be a comprehensive factual answer covering all sub-questions.
Include specific data points, numbers, and recommendations WITH their source citations.
Keep verification_notes brief - just list what was cross-checked or any resolved contradictions."""


def synthesizer_node(state: DeepResearchState) -> dict:
    print("Running synthesizer")

    llm = ModelRegistry.get("synthesizer")
    user_query = state["user_query"]
    worker_reports = state.get("worker_reports", [])
    retry_count = state.get("retry_count", 0)

    if not worker_reports:
        print("Synthesizer: no worker reports")
        return {
            "synthesis": {
                "synthesized_answer": "No research data available to answer this question.",
                "overall_confidence": "low",
                "verification_notes": "No worker reports received.",
                "sources_used": [],
                "needs_retry": False,
            },
            "needs_retry": False,
        }

    formatted_reports = ""
    for i, report in enumerate(worker_reports):
        formatted_reports += (
            f"\nReport {i+1}:\n"
            f"Sub-question: {report.get('sub_question', 'N/A')}\n"
            f"Answer found: {report.get('answer_found', False)}\n"
            f"Confidence: {report.get('confidence', 'low')}\n"
            f"Sources: {', '.join(report.get('sources', []))}\n"
            f"Findings:\n{report.get('findings', 'No findings')}\n"
        )

    user_prompt = (
        f"Original User Question: {user_query}\n\n"
        f"Worker Research Reports:{formatted_reports}"
    )

    try:
        structured_llm = llm.with_structured_output(SynthesisResult)
        result = structured_llm.invoke([
            {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
    except Exception as e:
        # print(f"Structured synthesis failed, using fallback: {type(e).__name__}")
        try:
            fallback_prompt = (
                SYNTHESIZER_SYSTEM_PROMPT
                + "\n\nRespond with ONLY a valid JSON object, no markdown, no explanation. "
                "Keys: synthesized_answer (string), overall_confidence (high/medium/low string), "
                "verification_notes (string), sources_used (array of strings), "
                "needs_retry (boolean true or false), retry_questions (array of strings or null). "
                "Use actual JSON booleans (true/false), not strings."
            )
            raw_result = llm.invoke([
                {"role": "system", "content": fallback_prompt},
                {"role": "user", "content": user_prompt},
            ])
            text = (raw_result.content or "").strip()
            extracted = _extract_first_json_object(text) or text

            # Common LLM quirks: quoted booleans/nulls.
            extracted = extracted.replace(': "false"', ': false').replace(': "true"', ': true')
            extracted = extracted.replace(': "null"', ': null').replace(': "None"', ': null')

            parsed = _json_loads_relaxed(extracted)
            if isinstance(parsed, str):
                # Some models double-encode JSON as a string.
                parsed = _json_loads_relaxed(parsed)

            if isinstance(parsed.get("needs_retry"), str):
                parsed["needs_retry"] = parsed["needs_retry"].strip().lower() == "true"
            if isinstance(parsed.get("retry_questions"), str):
                parsed["retry_questions"] = None

            # Normalize common schema mismatches.
            parsed["sources_used"] = _coerce_str_list(parsed.get("sources_used"))
            if parsed.get("retry_questions") is not None:
                parsed["retry_questions"] = _coerce_str_list(parsed.get("retry_questions"))

            result = SynthesisResult(**parsed)
        except Exception as fallback_e:
            # print(f"Fallback synthesis also failed: {fallback_e}")
            all_findings = "\n\n".join(
                r.get("findings", "") for r in worker_reports if r.get("findings")
            )
            all_sources = []
            for r in worker_reports:
                all_sources.extend(r.get("sources", []))

            retry_questions = _retry_from_worker_reports(worker_reports)
            allow_retry = retry_count < 2 and bool(retry_questions)
            result = SynthesisResult(
                synthesized_answer=all_findings[:3000],
                overall_confidence="low" if allow_retry else "medium",
                verification_notes="Used raw worker findings directly (JSON synthesis unavailable).",
                sources_used=list(set(all_sources)),
                needs_retry=allow_retry,
                retry_questions=retry_questions if allow_retry else None,
            )


    should_retry = result.needs_retry and retry_count < 2 and bool(result.retry_questions)

    print(f"Synthesizer: confidence={result.overall_confidence} | retry={should_retry}")

    return {
        "synthesis": result.dict(),
        "needs_retry": should_retry,
        "sources": result.sources_used,
    }
