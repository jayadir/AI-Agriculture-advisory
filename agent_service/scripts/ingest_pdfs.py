import os
import sys

# Add the root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

# Force settings specifically to use BGE since you requested it
os.environ["EMBEDDING_MODEL"] = "bge"

# Import exactly what your app uses
from app.rag.embeddings import get_embedder

def ingest_pdfs():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    
    # 1. Find all PDFs
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDFs found in data folder.")
        return

    documents = []
    
    # 2. Load the PDFs robustly
    print(f"Found {len(pdf_files)} PDFs. Loading...")
    for pdf in pdf_files:
        pdf_path = os.path.join(data_dir, pdf)
        print(f" - Loading: {pdf}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        # Ensure metadata has 'source' so RAG can reference it properly
        for doc in docs:
            doc.metadata["source"] = pdf
        documents.extend(docs)
        
    print(f"Loaded {len(documents)} total pages.")
    
    # 3. Proper chunking strategy
    # BGE-small-en-v1.5 has an ~512 token context limit which is roughly 2000 chars.
    # 1000 chunk size gives it enough room to capture context, with 150 overlap for continuity.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split pages into {len(chunks)} chunks.")
    
    # 4. Get the exact same BGE embedder
    embedder = get_embedder()
    
    vector_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../artifacts/vector_db/agri_faiss_index"))
    
    # 5. Connect to existing FAISS or start a new one
    vectorstore = None
    if os.path.exists(vector_db_path):
        print(f"Found existing FAISS index at {vector_db_path}. Merging new docs...")
        vectorstore = FAISS.load_local(
            vector_db_path, 
            embedder,
            allow_dangerous_deserialization=True
        )
    else:
        print(f"No existing FAISS index found at {vector_db_path}. Creating a new one...")
        os.makedirs(os.path.dirname(vector_db_path), exist_ok=True)
        
    # Ingest in batches to show progress
    batch_size = 100
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding & Ingesting chunks"):
        batch = chunks[i:i+batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embedder)
        else:
            vectorstore.add_documents(batch)
        
    # 6. Save!
    vectorstore.save_local(vector_db_path)
    print(f"✅ Successfully ingested {len(chunks)} chunks into the FAISS Knowledge Base!")

if __name__ == "__main__":
    ingest_pdfs()
