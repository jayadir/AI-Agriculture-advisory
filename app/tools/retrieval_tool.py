from app.rag.engine import get_rag_engine
from langchain.tools import tool
import asyncio

@tool("retrieval_tool", args_schema={"query": str})
async def retrieval_tool(query: str):
    """ this tool is used to retrieve relevant documents from the local vector store based on the query provided by the user. """
    engine= await get_rag_engine()
    result = await engine.process(query)
    return result