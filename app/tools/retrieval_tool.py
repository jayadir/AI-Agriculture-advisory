from app.rag.engine import get_rag_engine
from langchain.tools import tool
import asyncio
from pydantic import BaseModel


class RetrievalToolInput(BaseModel):
    query: str

@tool("retrieval_tool", args_schema=RetrievalToolInput)
def retrieval_tool(query: str):
    """ this tool is used to retrieve relevant documents from the local vector store based on the query provided by the user. """
    async def _run() -> dict:
        engine = await get_rag_engine()
        return await engine.process(query)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        raise RuntimeError(
            "retrieval_tool was invoked from within a running event loop. "
            "This tool is currently implemented as a synchronous LangChain tool; "
            "invoke it from a sync context (LangGraph tool node) or refactor the agent to async." 
        )

    return asyncio.run(_run())