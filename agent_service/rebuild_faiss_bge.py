"""
FAISS Index Rebuild Script for BGE Embeddings
==============================================
This script rebuilds the FAISS vector database using BGE-small-en-v1.5 embeddings.

Usage:
    1. Set environment variable: $env:EMBEDDING_MODEL="bge"
    2. Run: python rebuild_faiss_bge.py
    
The script will:
    - Load the existing FAISS index (if available)
    - Extract all documents
    - Re-embed using BGE model (384-dim)
    - Save new index to artifacts/vector_db/agri_faiss_index_bge
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set embedding model to BGE BEFORE importing anything
os.environ["EMBEDDING_MODEL"] = "bge"

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.rag.embeddings import get_embedder
import time
import json
from tqdm import tqdm

VECTOR_DIR = "artifacts/vector_db"
OLD_INDEX_PATH = f"{VECTOR_DIR}/agri_faiss_index"
NEW_INDEX_PATH = f"{VECTOR_DIR}/agri_faiss_index_bge"
BACKUP_PATH = f"{VECTOR_DIR}/agri_faiss_index_backup_{int(time.time())}"

def extract_documents_from_faiss(index_path):
    """Extract all documents from existing FAISS index"""
    print(f"📖 Loading existing FAISS index from {index_path}...")
    
    # Load with a temporary embedder (we just need the documents)
    from app.rag.embeddings import JinaEmbedder
    temp_embedder = JinaEmbedder()
    
    try:
        vectorstore = FAISS.load_local(
            index_path,
            temp_embedder,
            allow_dangerous_deserialization=True
        )
        
        # Access the docstore to get all documents
        docstore = vectorstore.docstore
        docs = []
        
        # FAISS uses index_to_docstore_id mapping
        if hasattr(vectorstore, 'index_to_docstore_id'):
            total_docs = len(vectorstore.index_to_docstore_id)
            print(f"⏳ Extracting {total_docs} documents...")
            
            with tqdm(total=total_docs, desc="Extracting docs", unit="doc") as pbar:
                for idx in vectorstore.index_to_docstore_id.values():
                    doc = docstore.search(idx)
                    if doc:
                        docs.append(doc)
                    pbar.update(1)
        
        print(f"✅ Extracted {len(docs)} documents from existing index")
        return docs
        
    except Exception as e:
        print(f"❌ Error loading existing index: {e}")
        print("📝 You may need to provide documents manually")
        return None

def load_documents_from_jsonl(jsonl_path):
    """Load documents from JSONL file"""
    print(f"📖 Loading documents from {jsonl_path}...")
    documents = []
    
    # Count total lines first for progress bar
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f if line.strip())
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        with tqdm(total=total_lines, desc="Loading docs", unit="doc") as pbar:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Adjust field names based on your JSONL structure
                    content = data.get('text', data.get('content', data.get('passage', '')))
                    metadata = {k: v for k, v in data.items() if k not in ['text', 'content', 'passage']}
                    documents.append(Document(page_content=content, metadata=metadata))
                    pbar.update(1)
    
    print(f"✅ Loaded {len(documents)} documents from JSONL")
    return documents

def rebuild_faiss_with_bge(documents):
    """Rebuild FAISS index using BGE embeddings with progress tracking"""
    if not documents:
        print("❌ No documents provided. Cannot rebuild index.")
        return False
    
    print(f"\n🔧 Rebuilding FAISS index with BGE-Small-EN-v1.5 (384-dim)...")
    print(f"   Total documents: {len(documents)}")
    
    # Get BGE embedder
    print("\n📥 Loading BGE embedder...")
    embedder = get_embedder()
    
    # Estimate time (rough estimate: ~0.02s per doc on CPU, ~0.005s on GPU)
    estimated_time = len(documents) * 0.02
    print(f"⏱️  Estimated time: {estimated_time:.1f}s ({estimated_time/60:.1f} minutes)")
    
    # Create new FAISS index with progress tracking
    print("\n⏳ Embedding documents and building FAISS index...")
    print("   (This creates embeddings for all documents and builds the vector index)")
    
    start_time = time.time()
    
    # Batch embedding with progress tracking
    batch_size = 32
    all_embeddings = []
    
    print(f"\n📊 Processing {len(documents)} documents in batches of {batch_size}...")
    with tqdm(total=len(documents), desc="Embedding docs", unit="doc") as pbar:
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_texts = [doc.page_content for doc in batch]
            
            # Embed batch
            batch_embeddings = embedder.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            pbar.update(len(batch))
            
            # Show elapsed and estimated remaining time
            elapsed = time.time() - start_time
            docs_done = i + len(batch)
            if docs_done > 0:
                rate = elapsed / docs_done
                remaining = rate * (len(documents) - docs_done)
                pbar.set_postfix({
                    'elapsed': f'{elapsed:.1f}s',
                    'remaining': f'{remaining:.1f}s',
                    'rate': f'{1/rate:.1f} docs/s'
                })
    
    embedding_time = time.time() - start_time
    print(f"✅ Embedding complete in {embedding_time:.1f}s ({embedding_time/60:.1f} minutes)")
    print(f"   Average: {embedding_time/len(documents)*1000:.1f}ms per document")
    
    # Build FAISS index
    print("\n🔨 Building FAISS index structure...")
    vectorstore = FAISS.from_documents(documents, embedder)
    index_time = time.time() - start_time - embedding_time
    print(f"✅ Index built in {index_time:.1f}s")
    
    # Save the new index
    print(f"\n💾 Saving index to {NEW_INDEX_PATH}...")
    save_start = time.time()
    vectorstore.save_local(NEW_INDEX_PATH)
    save_time = time.time() - save_start
    print(f"✅ Index saved in {save_time:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n✅ Successfully rebuilt FAISS index with BGE embeddings!")
    print(f"   Location: {NEW_INDEX_PATH}")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    
    return True

def main():
    print("=" * 70)
    print("FAISS Index Rebuild Script - BGE Embeddings")
    print("=" * 70)
    
    # Check if old index exists
    if os.path.exists(OLD_INDEX_PATH):
        print(f"\n✓ Found existing FAISS index at {OLD_INDEX_PATH}")
        
        # Try to extract documents
        documents = extract_documents_from_faiss(OLD_INDEX_PATH)
        
        if documents:
            # Rebuild with BGE
            success = rebuild_faiss_with_bge(documents)
            
            if success:
                print("\n" + "=" * 70)
                print("✅ REBUILD COMPLETE!")
                print("=" * 70)
                print("\nNext steps:")
                print(f"1. Backup old index (optional):")
                print(f"   Move-Item '{OLD_INDEX_PATH}' '{BACKUP_PATH}'")
                print(f"\n2. Replace old index with new one:")
                print(f"   Move-Item '{NEW_INDEX_PATH}' '{OLD_INDEX_PATH}'")
                print(f"\n3. Update your .env file:")
                print(f"   EMBEDDING_MODEL=bge")
                return True
        else:
            print("\n⚠️  Could not extract documents from existing index.")
    else:
        print(f"\n⚠️  No existing FAISS index found at {OLD_INDEX_PATH}")
    
    # Alternative: Load from JSONL
    print("\n📋 Alternative: Load documents from JSONL file")
    print("   Place your documents in a JSONL file and update the path below.")
    
    # Example JSONL paths to check
    jsonl_paths = [
        "notebooks/agri_synthetic_data.jsonl",
        "notebooks/final_written_queries.jsonl",
    ]
    
    for jsonl_path in jsonl_paths:
        if os.path.exists(jsonl_path):
            print(f"\n✓ Found JSONL file: {jsonl_path}")
            choice = input(f"   Load documents from this file? (y/n): ").strip().lower()
            
            if choice == 'y':
                documents = load_documents_from_jsonl(jsonl_path)
                if documents:
                    success = rebuild_faiss_with_bge(documents)
                    if success:
                        print("\n" + "=" * 70)
                        print("✅ REBUILD COMPLETE!")
                        print("=" * 70)
                        return True
    
    print("\n" + "=" * 70)
    print("❌ Rebuild incomplete. Please provide document source.")
    print("=" * 70)
    return False

if __name__ == "__main__":
    main()
