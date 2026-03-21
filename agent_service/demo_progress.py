"""
Quick test to demonstrate progress tracking in rebuild script
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set embedding model to BGE
os.environ["EMBEDDING_MODEL"] = "bge"

from langchain_core.documents import Document
from tqdm import tqdm
import time

def simulate_rebuild_progress():
    """Simulate the rebuild process to show progress bars"""
    
    print("=" * 70)
    print("FAISS Rebuild Progress Demo")
    print("=" * 70)
    
    # Simulate document extraction
    print("\n📖 Step 1: Extracting documents from existing index")
    num_docs = 100
    docs = []
    
    with tqdm(total=num_docs, desc="Extracting docs", unit="doc") as pbar:
        for i in range(num_docs):
            # Simulate extraction work
            time.sleep(0.01)
            docs.append(Document(page_content=f"Document {i}", metadata={"id": i}))
            pbar.update(1)
    
    print(f"✅ Extracted {len(docs)} documents")
    
    # Simulate embedding
    print("\n📊 Step 2: Embedding documents with BGE model")
    batch_size = 10
    start_time = time.time()
    
    with tqdm(total=len(docs), desc="Embedding docs", unit="doc") as pbar:
        for i in range(0, len(docs), batch_size):
            # Simulate embedding work (BGE is fast!)
            time.sleep(0.05)
            
            docs_done = min(i + batch_size, len(docs))
            pbar.update(batch_size if i + batch_size <= len(docs) else len(docs) - i)
            
            # Update stats
            elapsed = time.time() - start_time
            rate = docs_done / elapsed if elapsed > 0 else 0
            remaining = (len(docs) - docs_done) / rate if rate > 0 else 0
            
            pbar.set_postfix({
                'elapsed': f'{elapsed:.1f}s',
                'remaining': f'{remaining:.1f}s',
                'rate': f'{rate:.1f} docs/s'
            })
    
    embedding_time = time.time() - start_time
    print(f"✅ Embedding complete in {embedding_time:.1f}s")
    print(f"   Average: {embedding_time/len(docs)*1000:.1f}ms per document")
    
    # Simulate index building
    print("\n🔨 Step 3: Building FAISS index")
    for i in tqdm(range(20), desc="Building index", unit="step"):
        time.sleep(0.05)
    
    print("✅ Index built")
    
    # Simulate saving
    print("\n💾 Step 4: Saving index to disk")
    for i in tqdm(range(10), desc="Saving index", unit="step"):
        time.sleep(0.05)
    
    print("✅ Index saved")
    
    total_time = time.time() - start_time
    print(f"\n" + "=" * 70)
    print("✅ REBUILD COMPLETE!")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"\nThis is what you'll see when running rebuild_faiss_bge.py")

if __name__ == "__main__":
    simulate_rebuild_progress()
