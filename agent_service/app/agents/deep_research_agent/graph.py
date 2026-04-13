from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from app.core.config import settings
from app.agents.deep_research_agent.state import DeepResearchState
from app.agents.deep_research_agent.nodes.planner import planner_node
from app.agents.deep_research_agent.nodes.worker import worker_node
from app.agents.deep_research_agent.nodes.synthesizer import synthesizer_node
from app.agents.deep_research_agent.nodes.responder import responder_node
from app.agents.deep_research_agent.edges import route_after_planner, route_after_synthesis


def _get_checkpointer() -> Optional[object]:
    mongo_url = getattr(settings, "MONGO_URL", None)
    db_name = getattr(settings, "DB_NAME", None)
    if not mongo_url or not db_name:
        return MemorySaver()
    try:
        from pymongo import MongoClient
        from langgraph.checkpoint.mongodb import MongoDBSaver
        client = MongoClient(mongo_url)
        return MongoDBSaver(client=client, db_name=db_name)
    except Exception:
        return MemorySaver()


def create_deep_research_agent():
    workflow = StateGraph(DeepResearchState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("worker", worker_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("responder", responder_node)

    workflow.set_entry_point("planner")

    workflow.add_conditional_edges("planner", route_after_planner)
    workflow.add_edge("worker", "synthesizer")
    workflow.add_conditional_edges(
        "synthesizer",
        route_after_synthesis,
        {
            "planner": "planner",
            "responder": "responder",
        },
    )
    workflow.add_edge("responder", END)

    return workflow.compile(checkpointer=_get_checkpointer())


def chat_with_agent(user_id: str, query: str, chat_history: list = None) -> Dict[str, Any]:
    print(f"\n{'='*60}\nDeep Research Agent Processing Query\n{'='*60}\n")

    agent = create_deep_research_agent()
    config = {"configurable": {"thread_id": user_id}}

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "user_query": query,
        "chat_history": chat_history or [],
        "route": "",
        "research_plan": None,
        "worker_reports": [],
        "synthesis": None,
        "needs_retry": False,
        "retry_count": 0,
        "final_response": "",
        "sources": [],
    }

    if chat_history:
        print(f"Using {len(chat_history) // 2} previous turns for context")

    final_response = ""
    for event in agent.stream(initial_state, config=config):
        for node_name, node_output in event.items():
            if node_name != "__end__":
                print(f"Completed: {node_name}")
            if isinstance(node_output, dict) and node_output.get("final_response"):
                final_response = node_output["final_response"]

    if not final_response:
        final_response = "I could not generate a response. Please try again."

    print(f"\n{'='*60}\nDeep Research Agent Complete\n{'='*60}\n")

    return {"user_id": user_id, "response": final_response}
