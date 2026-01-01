import json
import re
import os
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from pydantic import BaseModel,Field, ValidationError
from dotenv import load_dotenv
load_dotenv()
class DataQualityResponse(BaseModel):
    is_relavant: bool = Field(..., description="Indicates if the context is relevant to the query.")

SYSTEM_PROMPT = (
    "You are Agri-Brain, a helpful agriculture advisory assistant. "
    "Answer concisely and factually. If context is provided, prefer it; "
    "otherwise use general knowledge. If unsure, say so and suggest next steps."
)
model=None
def _get_model() :
    global model
    model_name = os.getenv("LLM_MODEL")
    if os.getenv("LLM_PROVIDER","GROQ")=="GROQ":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GROQ_API_KEY in environment.")
        # Default small instruct model; adjust as needed
        # global model
        if model is None:
            model = ChatGroq(api_key=api_key, model_name=model_name)
        return model
    elif os.getenv("LLM_PROVIDER")=="GOOGLE":
        google_api_key=os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY in environment.")
        # global model
        if model is None:
            model = ChatGoogleGenerativeAI(google_api_key=google_api_key, model="gemini-pro")
        return model
    elif os.getenv("LLM_PROVIDER")=="OLLAMA":
        # global model
        if model is None:
            model = ChatOllama(model=model_name)
        return model

async def generate_response(query: str, context: Optional[str] = None) -> str:
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
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate


async def classify_data_quality(query: str, context: str) -> bool:
    llm = _get_model()
    structured_llm = llm.with_structured_output(DataQualityResponse)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at evaluating the relevance of context to user queries in the agriculture domain. "
                   "Given a user query and a context passage, determine if the context is relevant to answering the query. "
                   "Respond with a JSON object containing a single boolean field 'is_relevant'."),
        ("human", "Question: {question}\n\nContext:\n{context}\n\n"
                  "Based on the above, is the context relevant to answering the question?")
    ])

    chain = prompt | structured_llm
    inputs = {"question": query, "context": context}
    try:
        response = await chain.ainvoke(inputs)
        return response.is_relevant
    except Exception as e:
        print("Structured output failed", e)
        return False

