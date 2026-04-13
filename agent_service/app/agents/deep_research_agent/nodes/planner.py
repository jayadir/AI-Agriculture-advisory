import json
from app.agents.deep_research_agent.model_registry import ModelRegistry
from app.agents.deep_research_agent.models import PlanDecision
from app.agents.deep_research_agent.state import DeepResearchState


PLANNER_SYSTEM_PROMPT = """You are a research planner for an Indian agriculture advisory system.

Given the user's question and conversation history, you must decide:

1. If the question is a greeting, thank-you, casual remark, or a simple follow-up that
   needs no factual data lookup, set route to "direct" and provide a short friendly
   direct_response in simple Indian English suitable for a farmer.

2. If the question needs factual agricultural information, set route to "research" and
   decompose it into 1 to 3 focused sub-questions that can each be researched independently.

For research routes:
- Each sub-question must be specific enough to retrieve targeted documents.
- Append India-specific context such as state, region, crop, or season to each sub-question.
- Choose search_strategy per sub-question: "web_only" or "kb_then_web".
    Use "web_only" for branded products, dosage schedules, and market/label questions.
    Use "kb_then_web" for everything else since the knowledge base may be incomplete.
- Provide india_localization per sub-question with region or crop context.
- Provide worker_instructions per sub-question, detailing exactly what the worker should
  look for, specific metrics (like cost, yield, dosages), and what reasoning to apply.

CRITICAL — DEDICATED MEASUREMENT WORKER:
- If the user's question relates to crops, soil, pests, diseases, fertilizers, or pesticides, YOU MUST dedicate ONE of the 3 sub-questions purely to extracting EXACT NUMERICAL DATA.
- Example sub_question: "What are the exact measurements, dosages, and numerical data for [topic]?"
- Example worker_instructions: "CRITICAL: Extract exact numbers, strict dosages, spacing, timing, and quantities."
- Do NOT add a measurement sub-question if the topic is purely administrative (e.g., PM-KISAN eligibility, schemes, bank loans).

CRITICAL — SEARCH KEYWORD GENERATION:
- For each sub-question, you MUST also provide a search_keywords string.
- Farmers ask questions in simple language. You must TRANSLATE their words into
  academic/extension terminology that search engines can match to official documents.
  Examples:
    Farmer says: "my rice has spots"     → search_keywords: "rice blast disease chemical control fungicide dosage TNAU recommendation"
    Farmer says: "how much DAP for rice" → search_keywords: "DAP fertilizer application rate rice kg/ha ICAR recommendation Tamil Nadu"
    Farmer says: "my crop is turning yellow" → search_keywords: "rice yellowing leaf nutrient deficiency diagnosis management India"
- Prefer official/extension terminology when relevant, but do not restrict to specific domains.
- Include the scientific/official name of diseases, pests, or crops when you can infer them.

Do not over-decompose simple single-topic questions. A question like "how to grow tomato"
should be a single sub-question, not three."""


def _build_history_context(chat_history):
    if not chat_history:
        return ""
    lines = []
    for msg in chat_history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    if not lines:
        return ""
    return "Previous Conversation:\n" + "\n".join(lines) + "\n\n"


def planner_node(state: DeepResearchState) -> dict:
    print("Running planner")

    llm = ModelRegistry.get("planner")
    user_query = state["user_query"]
    chat_history = state.get("chat_history", [])
    history_context = _build_history_context(chat_history)

    retry_questions = None
    if state.get("needs_retry") and state.get("synthesis"):
        synthesis = state["synthesis"]
        retry_questions = synthesis.get("retry_questions")

    if retry_questions:
        user_prompt = (
            f"{history_context}Original Question: {user_query}\n\n"
            f"The following sub-questions FAILED to find exact data in the previous attempt:\n"
            f"{json.dumps(retry_questions)}\n\n"
            f"Create a NEW research plan for these questions only. "
            f"CRITICAL: You MUST generate completely DIFFERENT `search_keywords` than your first attempt. Use new synonyms, alternative official terms, or broader concepts to ensure we hit new documents."
        )
    else:
        user_prompt = f"{history_context}User Question: {user_query}"

    try:
        structured_llm = llm.with_structured_output(PlanDecision)
        result = structured_llm.invoke([
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
    except Exception as e:
        print(f"Structured planner failed: {e}, using fallback")
        try:
            fallback_prompt = (
                PLANNER_SYSTEM_PROMPT
                + "\n\nRespond with a JSON object with keys: route, direct_response, "
                "sub_questions, search_strategies, india_localization, worker_instructions"
            )
            raw_result = llm.invoke([
                {"role": "system", "content": fallback_prompt},
                {"role": "user", "content": user_prompt},
            ])
            parsed = json.loads(raw_result.content)
            result = PlanDecision(**parsed)
        except Exception:
            result = PlanDecision(
                route="research",
                sub_questions=[user_query],
                search_strategies=["kb_then_web"],
                india_localization=["India"],
                worker_instructions=["Find comprehensive information on this topic."],
            )

    if result.route == "direct":
        print(f"Route: direct")
        return {
            "route": "direct",
            "final_response": result.direct_response or "Hello, how can I help you with your farming today?",
            "research_plan": None,
        }

    if not result.sub_questions:
        result.sub_questions = [user_query]
        result.search_strategies = ["kb_then_web"]
        result.india_localization = ["India"]

    while len(result.search_strategies or []) < len(result.sub_questions):
        result.search_strategies = (result.search_strategies or []) + ["kb_then_web"]
    while len(result.india_localization or []) < len(result.sub_questions):
        result.india_localization = (result.india_localization or []) + ["India"]
    while len(result.worker_instructions or []) < len(result.sub_questions):
        result.worker_instructions = (result.worker_instructions or []) + ["Extract relevant facts and figures."]
    while len(result.search_keywords or []) < len(result.sub_questions):
        result.search_keywords = (result.search_keywords or []) + [result.sub_questions[len(result.search_keywords or [])]]

    print(f"Route: research | Sub-questions: {len(result.sub_questions)}")
    for i, sq in enumerate(result.sub_questions):
        print(f"  [{i+1}] {sq} ({result.search_strategies[i]})")
        print(f"       Search Keywords: {result.search_keywords[i]}")

    return {
        "route": "research",
        "research_plan": result.dict(),
        "retry_count": state.get("retry_count", 0) + (1 if retry_questions else 0),
    }
