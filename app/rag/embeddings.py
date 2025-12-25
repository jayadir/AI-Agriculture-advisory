from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
import torch
import gc
import os

# --- MEMORY CONFIG ---
# Set this to True to prevent GPU OOM crashes on laptops
FORCE_CPU = True 
class JinaEmbedder(Embeddings):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(JinaEmbedder, cls).__new__(cls)
            
            # 1. Determine Device
            if FORCE_CPU:
                cls._instance.device = "cpu"
                print("🛡️ OOM Protection: Forcing CPU mode for Embeddings.")
            else:
                cls._instance.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"⏳ Loading Jina Embeddings V3 on {cls._instance.device}...")

            # Force a safe attention backend on Windows/CPU.
            # This avoids rare rotary/flash attention shape mismatches.
            os.environ.setdefault("TRANSFORMERS_ATTENTION_IMPLEMENTATION", "eager")
            
            # 2. Aggressive Cleanup before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # 3. Load Model (prefer eager attention)
            try:
                cls._instance.model = SentenceTransformer(
                    "jinaai/jina-embeddings-v3",
                    trust_remote_code=True,
                    device=cls._instance.device,
                    model_kwargs={"attn_implementation": "eager"},
                )
            except TypeError:
                # Older Transformers/SentenceTransformers may not support model_kwargs.
                cls._instance.model = SentenceTransformer(
                    "jinaai/jina-embeddings-v3",
                    trust_remote_code=True,
                    device=cls._instance.device,
                )
            print(f"✅ Jina V3 Loaded Successfully")
            
        return cls._instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds documents (Batched to save RAM)"""
        # Batch size 1 is slow but safest for memory
        embeddings = self.model.encode(
            texts, 
            prompt_name="retrieval.passage", 
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=1 
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query"""
        embedding = self.model.encode(
            [text], 
            prompt_name="retrieval.query", 
            convert_to_numpy=True
        )[0]
        return embedding.tolist()

def get_embedder():
    return JinaEmbedder()