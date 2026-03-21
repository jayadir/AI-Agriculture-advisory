"""
Setup Verification Script
=========================
Verifies that all components are correctly configured for BGE or Jina embeddings.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_env_vars():
    """Check environment variables"""
    print("\n" + "="*70)
    print("1. ENVIRONMENT VARIABLES")
    print("="*70)
    
    embedding_model = os.getenv("EMBEDDING_MODEL", "jina")
    force_cpu = os.getenv("FORCE_CPU", "0")
    lign_enabled = os.getenv("LIGN_ENABLED", "1")
    
    print(f"✓ EMBEDDING_MODEL: {embedding_model}")
    print(f"✓ FORCE_CPU: {force_cpu}")
    print(f"✓ LIGN_ENABLED: {lign_enabled}")
    
    return embedding_model.lower()

def check_model_files(embedding_model):
    """Check if required model files exist"""
    print("\n" + "="*70)
    print("2. MODEL FILES")
    print("="*70)
    
    if embedding_model == "bge":
        expander_path = "artifacts/models/query_expander.pth"
        expected_dim = 384
    else:
        expander_path = "artifacts/models/deep_residual_expander.pt"
        expected_dim = 1024
    
    print(f"\nExpected embedding dimension: {expected_dim}D")
    print(f"Expected expander path: {expander_path}")
    
    if os.path.exists(expander_path):
        print(f"✅ Query expander found: {expander_path}")
        
        # Check file size
        size_mb = os.path.getsize(expander_path) / (1024 * 1024)
        print(f"   File size: {size_mb:.2f} MB")
    else:
        print(f"❌ Query expander NOT found: {expander_path}")
        return False
    
    # Check other model files
    other_models = [
        "artifacts/models/agri_selector_v1.pt",
        "artifacts/models/neural_router_msmarco.pt",
    ]
    
    print("\nOther model files:")
    for model_path in other_models:
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✓ {model_path} ({size_mb:.2f} MB)")
        else:
            print(f"⚠ {model_path} (not found)")
    
    return True

def check_faiss_index(embedding_model):
    """Check if FAISS index exists"""
    print("\n" + "="*70)
    print("3. FAISS INDEX")
    print("="*70)
    
    index_path = "artifacts/vector_db/agri_faiss_index"
    
    if os.path.exists(index_path):
        print(f"✅ FAISS index found: {index_path}")
        
        # Check files in the index
        files = os.listdir(index_path)
        print(f"   Files: {', '.join(files)}")
        
        # Try to check dimension (if possible)
        index_file = os.path.join(index_path, "index.faiss")
        if os.path.exists(index_file):
            size_mb = os.path.getsize(index_file) / (1024 * 1024)
            print(f"   Index size: {size_mb:.2f} MB")
            
            # Note: We can't easily check dimension without loading
            print(f"\n⚠️  Note: Ensure this index was built with {embedding_model.upper()} embeddings")
            if embedding_model == "bge":
                print(f"   Expected dimension: 384")
            else:
                print(f"   Expected dimension: 1024")
        
        return True
    else:
        print(f"❌ FAISS index NOT found: {index_path}")
        print(f"   Run rebuild_faiss_bge.py to create the index")
        return False

def test_import():
    """Test if modules can be imported"""
    print("\n" + "="*70)
    print("4. MODULE IMPORTS")
    print("="*70)
    
    try:
        from app.rag.embeddings import get_embedder
        print("✓ app.rag.embeddings")
        
        embedder = get_embedder()
        print(f"✓ Embedder loaded: {type(embedder).__name__}")
    except Exception as e:
        print(f"❌ Failed to import embeddings: {e}")
        return False
    
    try:
        from app.rag.query_expander import DeepResidualExpander
        print("✓ app.rag.query_expander (Jina)")
    except Exception as e:
        print(f"❌ Failed to import Jina expander: {e}")
    
    try:
        from app.rag.query_expander_bge import DeepResidualExpanderBGE
        print("✓ app.rag.query_expander_bge (BGE)")
    except Exception as e:
        print(f"❌ Failed to import BGE expander: {e}")
        return False
    
    try:
        from app.rag.engine import RAGEngine
        print("✓ app.rag.engine")
    except Exception as e:
        print(f"❌ Failed to import RAGEngine: {e}")
        return False
    
    return True

def test_embedder():
    """Test embedder with sample text"""
    print("\n" + "="*70)
    print("5. EMBEDDER TEST")
    print("="*70)
    
    try:
        from app.rag.embeddings import get_embedder
        
        embedder = get_embedder()
        
        # Test query embedding
        test_query = "What are the symptoms of wheat rust?"
        print(f"\nTest query: '{test_query}'")
        
        embedding = embedder.embed_query(test_query)
        print(f"✅ Embedding generated")
        print(f"   Dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        
        # Verify dimension
        expected_dim = 384 if os.getenv("EMBEDDING_MODEL", "jina").lower() == "bge" else 1024
        if len(embedding) == expected_dim:
            print(f"✅ Dimension matches expected: {expected_dim}")
        else:
            print(f"❌ Dimension mismatch! Expected {expected_dim}, got {len(embedding)}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Embedder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*70)
    print("BGE/JINA EMBEDDING SETUP VERIFICATION")
    print("="*70)
    
    results = []
    
    # Run checks
    embedding_model = check_env_vars()
    results.append(("Environment Variables", True))
    
    model_files_ok = check_model_files(embedding_model)
    results.append(("Model Files", model_files_ok))
    
    faiss_ok = check_faiss_index(embedding_model)
    results.append(("FAISS Index", faiss_ok))
    
    import_ok = test_import()
    results.append(("Module Imports", import_ok))
    
    embedder_ok = test_embedder()
    results.append(("Embedder Test", embedder_ok))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for check, status in results:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check}")
    
    all_ok = all(status for _, status in results)
    
    if all_ok:
        print("\n🎉 All checks passed! Your setup is ready.")
        print(f"\n   Embedding Model: {embedding_model.upper()}")
        print(f"   You can now start the service with: python -m app.main")
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")
        print("\n   Common fixes:")
        print("   1. Set EMBEDDING_MODEL environment variable")
        print("   2. Ensure model files exist in artifacts/models/")
        print("   3. Run rebuild_faiss_bge.py to create FAISS index")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
