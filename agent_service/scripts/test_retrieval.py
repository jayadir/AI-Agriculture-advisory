import os
import sys
import asyncio

# Add the root directory to path so imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force the embedding model to BGE to match your ingestion script
os.environ["EMBEDDING_MODEL"] = "bge"

from app.rag.engine import get_rag_engine

async def run_test():
    print("Initializing RAG Engine...")
    # This automatically loads FAISS index and the deep residual query expander
    engine = await get_rag_engine()
    
    query = input("\nEnter your test query (or press Enter for default): ")
    if not query.strip():
        query = "my Improved White Ponni rice crop is getting blast disease, what protection measures should I take?"
        print(f"Using default query: '{query}'")
        
    print("\nSearching knowledge base...")
    
    # Process runs the query expander, fetches from FAISS, and reranks
    result = await engine.process(query)
    
    # The RAGEngine internally prints candidates, but let's summarize the final output
    response_docs = result.get("response_docs", "")
    reranked = result.get("reranked_top_docs", [])
    
    print("\n" + "="*60)
    print("FINAL TEXT CONTEXT PROVIDED TO THE AGENT")
    print("="*60)
    if response_docs.strip():
        print(response_docs)
    else:
        print("No context returned.")
    print("="*60)
    
    print(f"\nFinal Selected Sources: {len(reranked)}")
    for doc in reranked:
        print(f" - [Rank {doc.get('rank')}] Score: {doc.get('score', 0):.4f} | Source: {doc.get('source')}")

if __name__ == "__main__":
    # Prevent known RuntimeError relating to asyncio on Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(run_test())
