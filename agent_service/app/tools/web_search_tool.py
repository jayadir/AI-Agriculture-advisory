from langchain_tavily import TavilySearch
from langchain.tools import tool as lc_tool
from dotenv import load_dotenv
load_dotenv()

tavily = TavilySearch(
    max_results=5,
    topic="general",
    # include_raw_content=True
)

@lc_tool("web_search")
def web_search(query: str) -> str:
    """Web search via Tavily (topic forced to 'general'). Returns titles + URLs."""
    result = tavily.invoke({"query": query, "topic": "general"})
    try:
        return "\n".join(
            f"- {item.get('title','No title')} — {item.get('url','')}" for item in (result or [])
        )
    except Exception:
        return str(result)