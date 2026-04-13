import anyio
import uuid
import re
import json
from typing import List, Dict

from tavily import TavilyClient
from langchain_core.documents import Document

from app.db.mongodb import get_database
from app.core.config import settings
from app.utils.text_cleaner import processing_chain
from app.models.kb_docs import CandidateKnowledge, CandidateMetadata
from app.agents.deep_research_agent.graph import create_deep_research_agent

tavily_client = TavilyClient()
URL_PATTERN = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*"


def batch_urls(urls, batch_size=20):
    urls = list(urls)
    for i in range(0, len(urls), batch_size):
        yield urls[i : i + batch_size]


async def learn_from_session(thread_id: str):
    print(f"Starting knowledge ingestion for Thread {thread_id}")

    config = {"configurable": {"thread_id": thread_id}}
    db = await get_database()
    candidate_collection = db["candidate_knowledge"]

    agent = create_deep_research_agent()
    state = agent.get_state(config=config)

    if not state.values:
        print(f"No interactions found for Thread {thread_id}.")
        return

    worker_reports = state.values.get("worker_reports", [])
    urls_to_scrape = set()

    for report in worker_reports:
        for source in report.get("sources", []):
            if (
                source.startswith("http")
                and "youtube.com" not in source
                and "facebook" not in source
            ):
                urls_to_scrape.add(source.rstrip(","))

    if not urls_to_scrape:
        print("No valid URLs found to scrape.")
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
            print(f"Saved {len(candidates_to_insert)} chunks to candidate knowledge")

    except Exception as e:
        print(f"Learning scraping error: {e}")
