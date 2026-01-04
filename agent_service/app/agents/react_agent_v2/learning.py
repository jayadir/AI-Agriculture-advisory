import anyio
import uuid
import re
import json
from typing import List, Dict
from tavily import TavilyClient
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage

from app.db.mongodb import get_database
from app.core.config import settings
from app.utils.text_cleaner import processing_chain
from app.models.kb_docs import CandidateKnowledge, CandidateMetadata
from app.agents.react_agent_v2.graph import create_agriculture_agent

tavily_client = TavilyClient()
URL_PATTERN = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*"

def extract_scored_urls(content: str) -> List[Dict]:
    results = []
    try:
        data = json.loads(content)
        if isinstance(data, list):
            for item in data:
                url = item.get("url")
                score = float(item.get("score", 0))
                if url:
                    results.append({"url": url.rstrip(","), "score": score})
        return results
    except Exception:
        pass

    urls = re.findall(URL_PATTERN, content)
    for url in urls:
        results.append({"url": url.rstrip(","), "score": 1.0})
    return results

def filter_urls_by_score(items: List[Dict], threshold: float = 0.7) -> List[str]:
    return [item["url"] for item in items if item["score"] >= threshold]

def batch_urls(urls, batch_size=20):
    urls = list(urls)
    for i in range(0, len(urls), batch_size):
        yield urls[i : i + batch_size]

def learn_from_session_bg(thread_id: str):
    anyio.from_thread.run(learn_from_session, thread_id)

async def learn_from_session(thread_id: str):
    print(f"Starting knowledge ingestion for Thread {thread_id} ")

    config = {"configurable": {"thread_id": thread_id}}
    db = await get_database()
    candidate_collection = db["candidate_knowledge"]

    agent = create_agriculture_agent()
    state = agent.get_state(config=config)

    if not state.values:
        print(f"No interactions found for Thread {thread_id}.")
        return

    messages = state.values.get("messages", [])
    urls_to_scrape = set()

    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "web_search":
            urls = extract_scored_urls(msg.content)
            urls = filter_urls_by_score(urls, threshold=0.7)
            for url in urls:
                if "youtube.com" not in url and "facebook" not in url:
                    urls_to_scrape.add(url.rstrip(","))

    if not urls_to_scrape:
        print("[Background] No valid URLs found to scrape.")
        return

    candidates_to_insert = []

    try:
        batched = list(batch_urls(urls_to_scrape))
        for url_batch in batched:
            response = tavily_client.extract(urls=url_batch)
            
            for result in response.get("results", []):
                content = result.get("raw_content", "")
                if not content or len(content) < 50:
                    continue

                doc = Document(
                    page_content=content[:20000],
                    metadata={
                        "source_url": result["url"],
                        "title": "Web Search Result",
                        "thread_id": thread_id,
                        "document_id": str(uuid.uuid4()),
                    },
                )

                chunks = processing_chain.invoke(doc.page_content)

                for i, chunk in enumerate(chunks):
                    chunk_doc = Document(
                        page_content=chunk,
                        metadata=doc.metadata | {"chunk_index": i, "chunk_id": str(uuid.uuid4())},
                    )

                    candidate = CandidateKnowledge(
                        page_content=chunk_doc.page_content,
                        metadata=CandidateMetadata(
                            source_url=chunk_doc.metadata["source_url"],
                            title=chunk_doc.metadata.get("title", "Unknown"),
                            thread_id=chunk_doc.metadata["thread_id"],
                            document_id=chunk_doc.metadata["document_id"],
                            chunk_index=chunk_doc.metadata["chunk_index"],
                            chunk_id=chunk_doc.metadata["chunk_id"],
                        ),
                        status="pending",
                    )
                    candidates_to_insert.append(candidate.dict(by_alias=True, exclude={"id"}))

        if candidates_to_insert:
            candidate_collection.insert_many(candidates_to_insert)
            print(f"--- [Background] Saved {len(candidates_to_insert)} chunks ---")

    except Exception as e:
        print(f"[Background] Scraping Error: {e}")