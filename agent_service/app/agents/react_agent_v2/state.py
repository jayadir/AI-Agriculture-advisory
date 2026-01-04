from typing import List, TypedDict
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """
    Represents the state of the agent during a single execution.
    """
    messages: List[BaseMessage]
    user_query: str
    search_query: str
    retrieval_done: bool
    retrieval_results: str
    web_search_done: bool
    web_search_results: str
    needs_web_search: bool
    final_response: str
    iteration_count: int