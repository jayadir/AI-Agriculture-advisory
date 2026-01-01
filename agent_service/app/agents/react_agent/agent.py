import anyio
import os
import uuid
import re
import json
from typing import List, Dict
from pydantic import BaseModel, Field
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.messages import ToolMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from tavily import TavilyClient

from app.services.llm import _get_model
from app.core.config import settings
from app.tools.retrieval_tool import retrieval_tool
from app.tools.web_search_tool import web_search as web_search_tool
from app.rag.embeddings import get_embedder
from app.utils.text_cleaner import processing_chain
from app.db.mongodb import get_database
from app.models.kb_docs import CandidateKnowledge,CandidateMetadata


load_dotenv()
tavily_client=TavilyClient()  
client = MongoClient(settings.MONGO_URL)
checkpointer=MongoDBSaver(client=client,db_name=settings.DB_NAME)
agent=None

class LLMResponse(BaseModel):
    response: str = Field(..., description="The LLM generated response text.")


def get_react_agent(tools):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", 
         """You are an agriculture expert assistant specialized in Indian farming conditions, crops, climate, pests, and practices.

            You have access to exactly two tools:

            retrieval_tool for pest and agriculture documents

            web_search tool for additional verified information

            You must ALWAYS follow this sequence:
            First, call retrieval_tool.
            If and only if the retrieved documents are missing, incomplete, or not relevant to the user query, then call the web_search tool and you are free to call web_search multiple times if needed.
            Never skip the retrieval_tool.

            You are strictly forbidden from answering using your own knowledge.
            If neither tool provides sufficient relevant information, you must clearly state that reliable information is not available.

            You must NEVER hallucinate, assume, infer, or generalize beyond the tool outputs.

            You must use ONLY information that is explicitly returned by the tools.
            All references must come directly from the tool outputs.

            You must use ONLY information relevant to INDIA.
            If a tool returns data related to other countries, regions, or climates, you must ignore it completely.

            If no India-specific information is found, state that clearly instead of answering.

            **Your final response must be in plain text only and it should be concise,straightforward direct answer to the question without any additional info**.
            Do not use markdown, formatting symbols, bullet points, or special styling.
            Do not include headings.
            Do not include emojis.
            Do not include any links and references in your final answer.
        """
        ),
        ("human", "{user_query}")
    ])
    llm=_get_model()
    # structured_llm=llm.with_structured_output(LLMResponse)
    global agent
    if agent is None:
        agent=create_react_agent(llm,tools,checkpointer=checkpointer)
    return agent, prompt_template

def chat_with_agent(user_id:str,query:str,thread_id:str=None):
    tools = [retrieval_tool, web_search_tool]
    agent, prompt_template = get_react_agent(tools)
    if thread_id is None:
        thread_id = str(uuid.uuid4())
        print(f"New thread created with id: {thread_id}")
    config = {"configurable": {"thread_id": thread_id}}
    final_response=""
    for event in agent.stream(
        {"messages": prompt_template.format_messages(user_query=query)},        config=config,
        stream_mode="values"
    ):
        msg=event["messages"][-1]
        event["messages"][-1].pretty_print()
        if msg.type=="ai":
            final_response=msg.content
    return {"user_id": user_id, "thread_id": thread_id, "response": final_response}



URL_PATTERN = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'

def extract_scored_urls(content: str) -> List[Dict]:
    results = []

    try:
        data = json.loads(content)
        if isinstance(data, list):
            for item in data:
                url = item.get("url")
                score = float(item.get("score", 0))
                if url:
                    results.append({
                        "url": url.rstrip(","),
                        "score": score
                    })
        return results
    except Exception:
        pass

    urls = re.findall(URL_PATTERN, content)
    for url in urls:
        results.append({
            "url": url.rstrip(","),
            "score": 1.0
        })

    return results

def filter_urls_by_score(items: List[Dict], threshold: float = 0.7) -> List[Dict]:
    return [
        item["url"] for item in items
        if item["score"] >= threshold
        
    ]

def batch_urls(urls, batch_size=20):
    urls = list(urls)
    for i in range(0, len(urls), batch_size):
        yield urls[i:i + batch_size]


def learn_from_session_bg(thread_id: str):
    anyio.from_thread.run(learn_from_session, thread_id)

async def learn_from_session(thread_id:str):
    print(f"--- [Background] Starting knowledge ingestion for Thread {thread_id} ---")  
    config = {"configurable": {"thread_id": thread_id}}
    db=await get_database()
    candidate_collection=db["candidate_knowledge"]
    if agent is None:
        print("Agent not initialized.")
        return
    state=agent.get_state(config=config)
    if not state.values:
        print(f"No interactions found for Thread {thread_id}.")
        return
    messages=state.values.get("messages",[])
    urls_to_scrape=set()
    for msg in messages:
        if isinstance(msg,ToolMessage) and msg.name=="web_search":
            urls= extract_scored_urls(msg.content)
            urls=filter_urls_by_score(urls,threshold=0.7)
            for url in urls:
                if "youtube.com" not in url and "facebook" not in url:
                    urls_to_scrape.add(url.rstrip(','))
    if not urls_to_scrape:
        print("[Background] No valid URLs found to scrape.")
        return
    print(f"[Background] Scraping {len(urls_to_scrape)} URLs: {urls_to_scrape}")
    
    new_docs=[]
    candidates_to_insert=[]
    try:
        batched_urls = list(batch_urls(urls_to_scrape, batch_size=20))

        print(f"[Background] Scraping {len(urls_to_scrape)} URLs in {len(batched_urls)} batches")

        for batch_idx, url_batch in enumerate(batched_urls, start=1):
            print(f"[Background] Processing batch {batch_idx}/{len(batched_urls)}")

            response = tavily_client.extract(urls=url_batch)

            for result in response.get("results", []):
                content = result.get("raw_content", "")
                if not content or len(content) < 50:
                    continue

                print(f"Extracted: {result['url'][:30]}... ({len(content)} chars)")

                doc = Document(
                    page_content=content[:20000],
                    metadata={
                        "source_url": result["url"],
                        "title": "Web Search Result",
                        "thread_id": thread_id,
                        "document_id": str(uuid.uuid4())
                    }
                )

                chunks = processing_chain.invoke(doc.page_content)

                for i, chunk in enumerate(chunks):
                    chunk_doc = Document(
                        page_content=chunk,
                        metadata=doc.metadata | {
                            "chunk_index": i,
                            "chunk_id": str(uuid.uuid4())
                        }
                    )

                    candidate = CandidateKnowledge(
                        page_content=chunk_doc.page_content,
                        metadata=CandidateMetadata(
                            source_url=chunk_doc.metadata["source_url"],
                            title=chunk_doc.metadata.get("title", "Unknown"),
                            thread_id=chunk_doc.metadata["thread_id"],
                            document_id=chunk_doc.metadata["document_id"],
                            chunk_index=chunk_doc.metadata["chunk_index"],
                            chunk_id=chunk_doc.metadata["chunk_id"]
                        ),
                        status="pending"
                    )

                    candidates_to_insert.append(
                        candidate.dict(by_alias=True, exclude={"id"})
                    )

                new_docs.append(doc)

        if candidates_to_insert:
            candidate_collection.insert_many(candidates_to_insert)
            print(f"--- [Background] Saved {len(candidates_to_insert)} chunks to Candidate Store (MongoDB) ---")
        else:
            print("[Background] No valid content chunks generated.")

    except Exception as e:
        print(f"[Background] Tavily Extract Error: {e}")
        return

            
            
        
                    
    
    
