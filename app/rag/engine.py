import torch
import os
from langchain_community.vectorstores import FAISS
from app.rag.residual_selector import ResidualSelector
from app.rag.embeddings import get_embedder

CONFIDENCE_THRESHOLD = 0.6
RETRIEVAL_K = 5

# Match the setting in embeddings.py
FORCE_CPU = True 

class RAGEngine:
    def __init__(self):
        # Ensure we use the same device as the embedder
        if FORCE_CPU:
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"🔌 Initializing RAG Engine on {self.device}...")
        
        self.embedder = get_embedder()
        
        # Load FAISS
        vector_db_path = "artifacts/vector_db/agri_faiss_index"
        if os.path.exists(vector_db_path):
            self.vectorstore = FAISS.load_local(
                vector_db_path, 
                self.embedder,
                allow_dangerous_deserialization=True
            )
            print(f"📚 FAISS Index Loaded")
        else:
            self.vectorstore = None
            print(f"⚠️ Vector DB not found at {vector_db_path}")

        # Load Model B
        self.model = ResidualSelector(input_dim=1024).to(self.device)
        model_path = "artifacts/models/agri_selector_v1.pt"
        
        if os.path.exists(model_path):
            # map_location ensures weights load to CPU if needed
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("🧠 Agri-Selector Weights Loaded")
        else:
            print(f"⚠️ Model B weights not found")

    async def process(self, query: str) -> dict:
        if not self.vectorstore:
            return {"response": "System Error: KB missing", "source": "error"}

        # A. Retrieve
        docs = self.vectorstore.similarity_search(query, k=RETRIEVAL_K)
        candidate_texts = [d.page_content for d in docs]
        
        if not candidate_texts:
            return {"response": "No data found.", "source": "local-empty"}

        # B. Embed & Score
        # Note: self.embedder.model handles the device logic internally now
        q_emb_tensor = self.embedder.model.encode(
            [query], prompt_name="retrieval.query", convert_to_tensor=True
        ).to(self.device)
        
        d_embs_tensor = self.embedder.model.encode(
            candidate_texts, prompt_name="retrieval.passage", convert_to_tensor=True
        ).to(self.device)
        
        q_expanded = q_emb_tensor.expand(d_embs_tensor.shape[0], -1)
        
        with torch.no_grad():
            scores = self.model(q_expanded, d_embs_tensor)
            
        best_score, best_idx = torch.max(scores, dim=0)
        best_score = float(best_score.item())
        best_text = candidate_texts[best_idx]
        
        print(f"📊 Score: {best_score:.4f}")

        if best_score >= CONFIDENCE_THRESHOLD:
            return {"response": best_text, "source": "local", "confidence": best_score}
        else:
            return {"response": f"⚠️ [Low Confidence] {best_text}", "source": "local-low", "confidence": best_score}

rag_engine = RAGEngine()

async def get_rag_engine():
    return rag_engine