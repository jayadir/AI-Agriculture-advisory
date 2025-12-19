import os
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

SYSTEM_PROMPT = (
    "You are Agri-Brain, a helpful agriculture advisory assistant. "
    "Answer concisely and factually. If context is provided, prefer it; "
    "otherwise use general knowledge. If unsure, say so and suggest next steps."
)

def _get_model() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment.")
    # Default small instruct model; adjust as needed
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    return ChatGroq(api_key=api_key, model_name=model_name)

def generate_response(query: str, context: Optional[str] = None) -> str:
    """Generate an assistant reply using LangChain + Groq.

    Args:
        query: user question.
        context: optional retrieved context from RAG.
    Returns:
        model string response.
    """
    llm = _get_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])
    chain = prompt | llm
    inputs = {"question": query, "context": context or "(no context)"}
    result = chain.invoke(inputs)
    return result.content.strip()
