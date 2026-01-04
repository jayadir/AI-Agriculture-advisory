import argparse
import uuid
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from app.core.config import settings
from app.agents.react_agent_v2.state import AgentState
from app.agents.react_agent_v2.nodes import (
    parse_query_node,
    retrieval_node,
    evaluate_retrieval_node,
    transform_query_node,  
    web_search_node,
    evaluate_search_node,
    generate_response_node,
)
from app.agents.react_agent_v2.edges import should_web_search, should_continue_search


def _get_checkpointer() -> Optional[object]:
    """Prefer MongoDBSaver if configured; fall back to in-memory for quick local tests."""
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

def create_agriculture_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("parse_query", parse_query_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("evaluate_retrieval", evaluate_retrieval_node)
    workflow.add_node("transform_query", transform_query_node) 
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("evaluate_search", evaluate_search_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.set_entry_point("parse_query")

    workflow.add_edge("parse_query", "retrieval")
    workflow.add_edge("retrieval", "evaluate_retrieval")

    workflow.add_conditional_edges(
        "evaluate_retrieval",
        should_web_search, 
        {
            "web_search": "transform_query", 
            "generate_response": "generate_response"
        },
    )

   
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "evaluate_search")
    workflow.add_conditional_edges(
        "evaluate_search",
        should_continue_search,
        {
            "web_search": "transform_query", 
            "generate_response": "generate_response"
        },
    )

    workflow.add_edge("generate_response", END)

    app = workflow.compile(checkpointer=_get_checkpointer())
    return app

def chat_with_agent(user_id: str, query: str, thread_id: str = None) -> Dict[str, Any]:
    if thread_id is None:
        thread_id = str(uuid.uuid4())
        print(f"New thread created with id: {thread_id}")

    config = {"configurable": {"thread_id": thread_id}}
    agent = create_agriculture_agent()

    initial_state = {"messages": [HumanMessage(content=query)], "user_query": query}

    print(f"\n{'='*60}\nAgriculture Agent Processing Query\n{'='*60}\n")

    final_state = None
    for state in agent.stream(initial_state, config=config):
        final_state = state
        for node_name in state.keys():
            if node_name != "__end__":
                print(f"Executing: {node_name}")

    final_response = final_state.get("generate_response", {}).get(
        "final_response", "No response generated"
    )

    print(f"\n{'='*60}\nAgent Execution Complete\n{'='*60}\n")

    return {"user_id": user_id, "thread_id": thread_id, "response": final_response}





