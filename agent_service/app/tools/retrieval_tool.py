import asyncio
import concurrent.futures

from langchain.tools import tool
from pydantic import BaseModel

from app.rag.engine import get_rag_engine


class RetrievalToolInput(BaseModel):
    query: str


def _run_async_safely(coro):
    """
    Run an async coroutine from any context:
    - If no event loop is running: use asyncio.run()
    - If an event loop IS running (LangGraph/FastAPI): submit to a fresh
      background thread that has its own loop, then block until done.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None or not loop.is_running():
        return asyncio.run(coro)

    # Running inside an existing loop (LangGraph node context).
    # Spin up a plain thread with its own event loop to avoid nesting.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


@tool("retrieval_tool", args_schema=RetrievalToolInput)
def retrieval_tool(query: str):
    """Retrieve relevant documents from the local vector store based on the query."""
    async def _run() -> dict:
        engine = await get_rag_engine()
        return await engine.process(query)

    return _run_async_safely(_run())
