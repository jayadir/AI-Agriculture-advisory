from langgraph.graph import END
from langgraph.types import Send
from app.agents.deep_research_agent.state import DeepResearchState


def route_after_planner(state: DeepResearchState):
    if state.get("route") == "direct":
        return END

    plan = state.get("research_plan")
    if not plan or not plan.get("sub_questions"):
        return END

    sends = []
    sub_questions = plan["sub_questions"]
    strategies = plan.get("search_strategies", [])
    localizations = plan.get("india_localization", [])
    instructions = plan.get("worker_instructions", [])
    search_keywords_list = plan.get("search_keywords", [])

    for i, sq in enumerate(sub_questions):
        strategy = strategies[i] if i < len(strategies) else "kb_then_web"
        localization = localizations[i] if i < len(localizations) else "India"
        instruction = instructions[i] if i < len(instructions) else "Extract relevant facts and figures."
        search_kw = search_keywords_list[i] if i < len(search_keywords_list) else sq
        
        sends.append(Send("worker", {
            "sub_question": sq,
            "strategy": strategy,
            "localization": localization,
            "worker_instruction": instruction,
            "search_keywords": search_kw,
            "user_query": state["user_query"],
        }))

    return sends


def route_after_synthesis(state: DeepResearchState):
    if state.get("needs_retry") and state.get("retry_count", 0) < 2:
        return "planner"
    return "responder"
