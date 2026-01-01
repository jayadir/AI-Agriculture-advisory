import os
from datetime import datetime
from typing import List, Dict, Any, Tuple

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from app.rag.embeddings import get_embedder
from app.services.llm import generate_response


VECTOR_DB_PATH = os.getenv(
    "VECTOR_DB_PATH", "artifacts/vector_db/agri_faiss_index"
)


def _get_or_create_vectorstore():
    embedder = get_embedder()
    if os.path.exists(VECTOR_DB_PATH):
        try:
            vs = FAISS.load_local(
                VECTOR_DB_PATH,
                embedder,
                allow_dangerous_deserialization=True,
            )
            return vs
        except Exception:
            # If corrupted, create a new one (caller may decide to remove old path)
            pass
    return FAISS.from_texts([""], embedder)  # seed empty store


def web_search(query: str, k: int = 5) -> List[str]:
    wrapper = DuckDuckGoSearchAPIWrapper()
    results = wrapper.results(query, max_results=k)
    urls = []
    for r in results:
        link = r.get("link") or r.get("href")
        if link:
            urls.append(link)
    return urls


def fetch_and_chunk(urls: List[str]) -> List[Dict[str, Any]]:
    loader = WebBaseLoader(urls)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ". "]
    )
    all_chunks = []
    now = datetime.utcnow()
    for d in docs:
        chunks = splitter.split_text(d.page_content or "")
        for idx, c in enumerate(chunks):
            all_chunks.append(
                {
                    "text": c,
                    "metadata": {
                        "source": d.metadata.get("source"),
                        "title": d.metadata.get("title"),
                        "chunk": idx,
                        "fetched_at": now.isoformat() + "Z",
                    },
                }
            )
    return all_chunks


def ingest_into_faiss(chunks: List[Dict[str, Any]]):
    if not chunks:
        return 0
    vs = _get_or_create_vectorstore()
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    vs.add_texts(texts=texts, metadatas=metadatas)
    # Save for persistence
    vs.save_local(VECTOR_DB_PATH)
    return len(texts)


def extract_structured_summaries(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Group by source URL
    by_source: Dict[str, List[str]] = {}
    for c in chunks:
        src = c["metadata"].get("source") or "unknown"
        by_source.setdefault(src, []).append(c["text"])

    outputs: List[Dict[str, Any]] = []
    for src, parts in by_source.items():
        joined = "\n\n".join(parts[:8])  # keep prompt size reasonable
        prompt_context = (
            "You will extract structured agriculture-relevant insights. "
            "Return strict JSON with keys: title, key_points (list), facts (list), summary."
        )
        response = generate_response(
            query=prompt_context,
            context=f"Source: {src}\n\nContent:\n{joined}",
        )
        outputs.append(
            {
                "source": src,
                "extracted": response,
            }
        )
    return outputs


def answer_with_web_context(query: str, k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
    urls = web_search(query, k=k)
    chunks = fetch_and_chunk(urls)
    ingest_into_faiss(chunks)
    # Build context for final answer
    context = "\n\n".join([c["text"] for c in chunks[:10]])
    answer = generate_response(query, context)
    summaries = extract_structured_summaries(chunks)
    return answer, summaries

async def invoke_web_pipeline(query:str):
    pass