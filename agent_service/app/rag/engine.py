import torch
import os
from langchain_community.vectorstores import FAISS
from app.rag.embeddings import get_embedder
from app.rag.query_expander import DeepResidualExpander
from app.rag.query_expander_bge import DeepResidualExpanderBGE
from app.rag.reranker import get_reranker
from app.utils.torch_numpy import tensor_to_numpy
from app.core.config import settings

RETRIEVAL_K = 10  # Retrieve 10 docs per query variant
MAX_CONTEXT_DOCS = settings.RERANKER_TOP_K  # Return top reranked docs for context

# Device selection: set `FORCE_CPU=1` in the environment to force CPU.
FORCE_CPU = os.getenv("FORCE_CPU", "0").strip().lower() in ("1", "true", "yes", "y", "on")

class RAGEngine:
    def __init__(self):
        if FORCE_CPU:
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"Initializing RAG Engine on {self.device}...")
        
        self.embedder = get_embedder()
        self.reranker = get_reranker() if settings.ENABLE_RERANKER else None
        
        # Load FAISS
        vector_db_path = "artifacts/vector_db/agri_faiss_index"
        if os.path.exists(vector_db_path):
            self.vectorstore = FAISS.load_local(
                vector_db_path, 
                self.embedder,
                allow_dangerous_deserialization=True
            )
            print(f"FAISS Index Loaded")
        else:
            self.vectorstore = None
            print(f"Vector DB not found at {vector_db_path}")

        # Dynamically load expander based on embedding model
        embedding_model = settings.EMBEDDING_MODEL.lower()
        if embedding_model == "bge":
            self.expander = DeepResidualExpanderBGE(input_dim=384).to(self.device)
            expander_path = "artifacts/models/query_expander.pth"
            print(f"Loading BGE Query Expander (384-dim)...")
        else:
            self.expander = DeepResidualExpander(input_dim=1024).to(self.device)
            expander_path = "artifacts/models/deep_residual_expander.pt"
            print(f"Loading Jina Query Expander (1024-dim)...")
        
        if os.path.exists(expander_path):
            state_dict = torch.load(expander_path, map_location=self.device)
            self.expander.load_state_dict(state_dict)
            self.expander.eval()
            print(f"Query Expander Weights Loaded from {expander_path}")
        else:
            print(f"Query Expander weights not found at {expander_path}")

    @staticmethod
    def _doc_preview(text: str, limit: int = 180) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return f"{compact[:limit - 3]}..."

    @staticmethod
    def _doc_source(metadata: dict) -> str:
        if not metadata:
            return "unknown"
        return (
            metadata.get("source")
            or metadata.get("title")
            or metadata.get("file_path")
            or metadata.get("url")
            or "unknown"
        )

    def _print_retrieval_candidates(self, candidates: list[dict], raw_hits: int):
        print("\n================ RETRIEVAL CANDIDATES ================")
        print(f"Raw hits: {raw_hits} | Unique candidates: {len(candidates)}")
        for index, candidate in enumerate(candidates, start=1):
            hit_summary = ", ".join(
                f"{hit['query_variant']}#{hit['rank']}" for hit in candidate["retrieval_hits"]
            )
            print(
                f"[{index:02d}] source={candidate['source']} | hits={hit_summary}\n"
                f"     {self._doc_preview(candidate['content'])}"
            )
        print("======================================================\n")

    def _print_reranked_docs(self, reranked_docs: list[dict], top_k: int):
        print("\n================ RERANKED CANDIDATES ================")
        print(f"Total scored candidates: {len(reranked_docs)} | Selected top_k: {top_k}")
        for candidate in reranked_docs:
            print(
                f"[#{candidate['rerank_rank']:02d}] score={candidate['rerank_score']:.4f} | source={candidate['source']}\n"
                f"     {self._doc_preview(candidate['content'])}"
            )
        print("=====================================================\n")

    async def process(self, query: str) -> dict:
        if not self.vectorstore:
            return {"response": "System Error: KB missing", "source": "error"}

        # Encode base query through the configured embedder so prompt handling
        # stays model-specific (for example, Jina vs BGE prompt formats).
        q_base_vector = self.embedder.embed_query(query)
        q_base_tensor = torch.tensor(
            [q_base_vector],
            dtype=torch.float32,
            device=self.device,
        )
        
        # Convert to float32 for stability
        if q_base_tensor.dtype in (torch.bfloat16, torch.float16):
            q_base_tensor = q_base_tensor.to(dtype=torch.float32)
        
        # Generate 4 query expansions
        with torch.no_grad():
            v_para, v_broad, v_tech, v_expl = self.expander(q_base_tensor)
        
        # Store all 5 query variants (original + 4 expansions)
        query_vectors = {
            "base": tensor_to_numpy(q_base_tensor),
            "para": tensor_to_numpy(v_para),
            "broad": tensor_to_numpy(v_broad), 
            "tech": tensor_to_numpy(v_tech),
            "expl": tensor_to_numpy(v_expl)
        }
        
        # Retrieve 10 documents per variant (5 variants * 10 docs = up to 50 docs)
        unique_candidates = {}
        raw_hits = 0
        
        for name, vec_np in query_vectors.items():
            results = self.vectorstore.similarity_search_by_vector(vec_np[0], k=RETRIEVAL_K)
            for rank, doc in enumerate(results, start=1):
                raw_hits += 1
                content = doc.page_content.strip()
                if not content:
                    continue

                existing = unique_candidates.get(content)
                if existing is None:
                    metadata = doc.metadata or {}
                    unique_candidates[content] = {
                        "content": content,
                        "metadata": metadata,
                        "source": self._doc_source(metadata),
                        "retrieval_hits": [{"query_variant": name, "rank": rank}],
                    }
                else:
                    existing["retrieval_hits"].append({"query_variant": name, "rank": rank})
        
        candidates = list(unique_candidates.values())

        if not candidates:
            return {"response": "No data found.", "source": "local-empty"}
        
        print(f"Retrieved {len(candidates)} unique documents from {len(query_vectors)} query variants")
        self._print_retrieval_candidates(candidates, raw_hits)

        if self.reranker:
            reranked_candidates = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=len(candidates),
            )
        else:
            reranked_candidates = candidates[:]
            for index, candidate in enumerate(reranked_candidates, start=1):
                candidate["rerank_score"] = 0.0
                candidate["rerank_rank"] = index

        self._print_reranked_docs(reranked_candidates, MAX_CONTEXT_DOCS)

        selected_candidates = reranked_candidates[:MAX_CONTEXT_DOCS]

        selected_docs = [candidate["content"] for candidate in selected_candidates]
        context = "\n--------------------------\n".join(selected_docs)
        
        return {
            "response_docs": context,
            "num_candidates": len(candidates),
            "reranked_top_docs": [
                {
                    "rank": candidate["rerank_rank"],
                    "score": candidate["rerank_score"],
                    "source": candidate["source"],
                    "preview": self._doc_preview(candidate["content"], limit=220),
                }
                for candidate in selected_candidates
            ],
        }
        
    def add_to_knowledge_base(self, documents):
        if not self.vectorstore:
            print("Vectorstore not initialized. Cannot add documents.")
            return
        self.vectorstore.add_documents(documents)
        self.vectorstore.save_local("artifacts/vector_db/agri_faiss_index")
        print(f"Added {len(documents)} documents to the knowledge base.")

rag_engine = RAGEngine()

async def get_rag_engine():
    return rag_engine