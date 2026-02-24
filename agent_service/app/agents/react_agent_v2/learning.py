import anyio
import uuid
import re
import json
from typing import List, Dict
from tavily import TavilyClient
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage, HumanMessage

from app.db.mongodb import get_database
from app.core.config import settings
from app.utils.text_cleaner import processing_chain
from app.utils.sliding_window_chunker import sliding_window_chunks, deduplicate_overlapping_chunks
from app.models.kb_docs import CandidateKnowledge, CandidateMetadata
from app.agents.react_agent_v2.graph import create_agriculture_agent
from app.models.lign_ranker import get_lign_ranker

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

    # Extract the actual Tavily search queries from web_search_results state
    # They are stored as: "--- Search 1 (query text) ---"
    SEARCH_HEADER_PATTERN = r"---\s*Search\s+\d+\s+\((.+?)\)\s*---"
    web_search_results = state.values.get("web_search_results", "")
    tavily_queries = re.findall(SEARCH_HEADER_PATTERN, web_search_results)

    if tavily_queries:
        lign_query = " ".join(tavily_queries)
        print(f"[Learning] Using {len(tavily_queries)} Tavily search queries for LIGN scoring: {tavily_queries}")
    else:
        # Fallback: use last 3 human messages if search queries not found
        lign_query = " ".join(
            msg.content for msg in messages
            if isinstance(msg, HumanMessage) and isinstance(msg.content, str)
        )[-300:]
        print(f"[Learning] No Tavily queries found, falling back to user messages for LIGN scoring.")

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

    # Load LIGN ranker once for the whole session
    lign_ranker = None
    if lign_query:
        try:
            lign_ranker = get_lign_ranker()
            print(f"[Learning] LIGN scoring enabled. Reference query: '{lign_query[:80]}...'")
        except Exception as e:
            print(f"[Learning] LIGN load failed, saving without scores: {e}")

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

                # Create overlapping chunks using sliding window (prevents boundary loss)
                windowed_chunks = sliding_window_chunks(
                    doc.page_content,
                    window_size=512,  # ~512 chars per chunk
                    overlap=128       # 128 char overlap
                )

                if not windowed_chunks:
                    continue

                # Score all windows in batch
                scored_windows = []
                if lign_ranker:
                    try:
                        chunk_texts = [chunk_text for chunk_text, _, _ in windowed_chunks]
                        scored = lign_ranker.batch_score(lign_query, chunk_texts)
                        
                        # Add position info for deduplication
                        for (chunk_text, start, end), (_, score) in zip(windowed_chunks, scored):
                            scored_windows.append((chunk_text, score, start, end))
                        
                        print(f"[Learning] Scored {len(scored_windows)} windows for {result['url']}")
                    except Exception as e:
                        print(f"[Learning] LIGN scoring failed for {result['url']}: {e}")
                        # Fallback: save without scores
                        for chunk_text, start, end in windowed_chunks:
                            scored_windows.append((chunk_text, None, start, end))
                else:
                    for chunk_text, start, end in windowed_chunks:
                        scored_windows.append((chunk_text, None, start, end))

                # Deduplicate overlapping chunks, keep highest-scored version
                LIGN_THRESHOLD = 0.4  # Only keep chunks with score >= 0.4
                deduplicated = deduplicate_overlapping_chunks(scored_windows, score_threshold=LIGN_THRESHOLD)
                
                print(f"[Learning] Kept {len(deduplicated)}/{len(scored_windows)} chunks after deduplication (threshold={LIGN_THRESHOLD})")

                for i, (chunk_text, score) in enumerate(deduplicated):
                    candidate = CandidateKnowledge(
                        page_content=chunk_text,
                        metadata=CandidateMetadata(
                            source_url=doc.metadata["source_url"],
                            title=doc.metadata.get("title", "Unknown"),
                            thread_id=doc.metadata["thread_id"],
                            document_id=doc.metadata["document_id"],
                            chunk_index=i,
                            chunk_id=str(uuid.uuid4()),
                            lign_score=score,
                        ),
                        status="pending",
                    )
                    candidates_to_insert.append(candidate.dict(by_alias=True, exclude={"id"}))

        if candidates_to_insert:
            await candidate_collection.insert_many(candidates_to_insert)
            scored_count = sum(1 for c in candidates_to_insert if c["metadata"].get("lign_score") is not None)
            print(f"--- [Background] Saved {len(candidates_to_insert)} chunks ({scored_count} with LIGN scores) ---")

    except Exception as e:
        print(f"[Background] Scraping Error: {e}")