from typing import TypedDict, Optional, List, Any, Annotated
import operator
from langchain_core.messages import BaseMessage


class DeepResearchState(TypedDict):
    messages: List[BaseMessage]
    user_query: str
    chat_history: List[Any]
    route: str
    research_plan: Optional[dict]
    worker_reports: Annotated[List[dict], operator.add]
    synthesis: Optional[dict]
    needs_retry: bool
    retry_count: int
    final_response: str
    sources: List[str]
