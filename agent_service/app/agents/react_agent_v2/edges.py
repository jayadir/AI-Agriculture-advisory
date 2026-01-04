from typing import Literal
from app.agents.react_agent_v2.state import AgentState

def should_web_search(state: AgentState) -> Literal["web_search", "generate_response"]:
    
    if state.get("needs_web_search"):
        return "web_search"
    return "generate_response"

def should_continue_search(state: AgentState) -> Literal["web_search", "generate_response"]:
    """Prevents infinite search loops by checking iteration count."""
    iteration_count = state.get("iteration_count", 0)
    max_iterations = 5

    if iteration_count >= max_iterations:
        print(f"Max iterations ({max_iterations}) reached")
        return "generate_response"

    if state.get("needs_web_search"):
        return "web_search"

    return "generate_response"