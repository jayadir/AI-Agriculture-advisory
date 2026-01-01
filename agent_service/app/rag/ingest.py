from langchain_community.vectorstores import FAISS
from app.rag.embeddings import get_embedder
import os
from langchain.schema import Document
import shutil
import time

VECTOR_DIR = "artifacts/vector_db"
LIVE_DIR = f"{VECTOR_DIR}/agri_faiss_index"
TMP_DIR = f"{VECTOR_DIR}/_tmp_update"

def prepare_tmp_index():
    if not os.path.exists(LIVE_DIR):
        raise RuntimeError("Live FAISS index does not exist")

    if os.path.exists(TMP_DIR):
        shutil.rmtree(TMP_DIR)

    shutil.copytree(LIVE_DIR, TMP_DIR)

def add_documents(documents):
    embedder= get_embedder()
    db= FAISS.load_local(
        TMP_DIR,
        embedder,
        allow_dangerous_deserialization=True
    )
    db.add_documents(documents)
    db.save_local(TMP_DIR)

def atomic_swap():
    backup = f"{LIVE_DIR}_backup_{int(time.time())}"
    os.rename(LIVE_DIR, backup)
    os.rename(TMP_DIR, LIVE_DIR)
    
def ingest(documents):
    prepare_tmp_index()
    add_documents(documents)
    atomic_swap()
    print("Incremental ingestion completed")
