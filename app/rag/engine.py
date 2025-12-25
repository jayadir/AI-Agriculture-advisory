import torch
import os
from langchain_community.vectorstores import FAISS
from app.rag.residual_selector import ResidualSelector
from app.rag.embeddings import get_embedder
from app.services.llm import generate_response
from app.rag.query_expander import DeepResidualExpander
from app.rag.router import NeuralRouterFusion
CONFIDENCE_THRESHOLD = 0.6
RETRIEVAL_K = 5

# Keep tool output small enough for hosted LLM token limits.
MAX_CONTEXT_DOCS = 3
MAX_CONTEXT_DOC_CHARS = 1200
MAX_CONTEXT_TOTAL_CHARS = 4500

# Match the setting in embeddings.py
FORCE_CPU = True 

class RAGEngine:
    def __init__(self):
        # Ensure we use the same device as the embedder
        if FORCE_CPU:
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"Initializing RAG Engine on {self.device}...")
        
        self.embedder = get_embedder()
        
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

        # Load Model B
        self.model = ResidualSelector(input_dim=1024).to(self.device)
        model_path = "artifacts/models/agri_selector_v1.pt"
        
        if os.path.exists(model_path):
            # map_location ensures weights load to CPU if needed
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print("Agri-Selector Weights Loaded")
        else:
            print(f"Model B weights not found")

        self.expander = DeepResidualExpander(input_dim=1024).to(self.device)
        expander_path = "artifacts/models/deep_residual_expander.pt"
        
        if os.path.exists(expander_path):
            state_dict = torch.load(expander_path, map_location=self.device)
            self.expander.load_state_dict(state_dict)
            self.expander.eval()
            print("Agri-Expander Weights Loaded")
        else:
            print(f"Model C weights not found")

        self.router = NeuralRouterFusion(input_dim=1024).to(self.device)
        router_path = "artifacts/models/neural_router_msmarco.pt"
        
        if os.path.exists(router_path):
            state_dict = torch.load(router_path, map_location=self.device)
            self.router.load_state_dict(state_dict)
            self.router.eval()
            print("Agri-Router Weights Loaded")
        else:
            print(f"Model D weights not found")

    async def process(self, query: str) -> dict:
        if not self.vectorstore:
            return {"response": "System Error: KB missing", "source": "error"}

        # A. Retrieve
        # docs = self.vectorstore.similarity_search(query, k=RETRIEVAL_K)
        # candidate_texts = [d.page_content for d in docs]
        
        # if not candidate_texts:
        #     return {"response": "No data found.", "source": "local-empty"}

        # # B. Embed & Score
        # # Note: self.embedder.model handles the device logic internally now
        # q_emb_tensor = self.embedder.model.encode(
        #     [query], prompt_name="retrieval.query", convert_to_tensor=True
        # ).to(self.device)
        
        # d_embs_tensor = self.embedder.model.encode(
        #     candidate_texts, prompt_name="retrieval.passage", convert_to_tensor=True
        # ).to(self.device)
        
        # q_expanded = q_emb_tensor.expand(d_embs_tensor.shape[0], -1)
        
        # with torch.no_grad():
        #     scores = self.model(q_expanded, d_embs_tensor)
            
        # best_score, best_idx = torch.max(scores, dim=0)
        # best_score = float(best_score.item())
        # best_text = candidate_texts[best_idx]
        
        # print(f"Score: {best_score:.4f}")

        # context = "\n\n".join(candidate_texts)
        # llm_answer = generate_response(query, context)
        # return {"response": llm_answer, "source": "llm", "confidence": best_score}
        
        q_base_tensor= self.embedder.model.encode(
            [query], prompt_name="retrieval.query", convert_to_tensor=True
        ).to(self.device)
        with torch.no_grad():
            v_para, v_broad, v_tech, v_expl = self.expander(q_base_tensor)
            scale_factor = 3.0 
    
            v_para = q_base_tensor + (v_para - q_base_tensor) * scale_factor
            v_broad = q_base_tensor + (v_broad - q_base_tensor) * scale_factor
            v_tech = q_base_tensor + (v_tech - q_base_tensor) * scale_factor
            v_expl = q_base_tensor + (v_expl - q_base_tensor) * scale_factor
        query_vectors={
            "base": q_base_tensor.cpu().numpy(),
            "para": v_para.cpu().numpy(),
            "broad": v_broad.cpu().numpy(), 
            "tech": v_tech.cpu().numpy(),
            "expl": v_expl.cpu().numpy()
        }
        # Debug: Check how different the variants are from the base
        # sim_tech = torch.nn.functional.cosine_similarity(q_base_tensor, v_tech)
        # print(f"Similarity (Base <-> Tech): {sim_tech.item():.4f}")
        # sim_para = torch.nn.functional.cosine_similarity(q_base_tensor, v_para)
        # print(f"Similarity (Base <-> Para): {sim_para.item():.4f}")
        # sim_broad = torch.nn.functional.cosine_similarity(q_base_tensor, v_broad)
        # print(f"Similarity (Base <-> Broad): {sim_broad.item():.4f}")
        # sim_expl = torch.nn.functional.cosine_similarity(q_base_tensor, v_expl)
        # print(f"Similarity (Base <-> Expl): {sim_expl.item():.4f}")
        
        unique_docs={}
        for name,vec_np in query_vectors.items():
            results=self.vectorstore.similarity_search_by_vector(vec_np[0], k=RETRIEVAL_K)
            
            # candidates=[doc.page_content for doc in results]
            # docs="\n--------------------\n".join(candidates)
            # print(f"Top-{RETRIEVAL_K} docs for '{name}' query vector:\n{docs}\n")
            for doc in results:
                if doc.page_content not in unique_docs:
                    unique_docs[doc.page_content] = doc
        candidate_texts = list(unique_docs.keys())
        if not candidate_texts:
            return {"response": "No data found.", "source": "local-empty"}
        print(f"Retrieved {len(candidate_texts)} unique documents.")
        d_embs_tensor = self.embedder.model.encode(
            candidate_texts, prompt_name="retrieval.passage", convert_to_tensor=True
        ).to(self.device)
        with torch.no_grad():
            s_base=torch.mm(q_base_tensor,d_embs_tensor.t())[0]
            s_para=torch.mm(v_para,d_embs_tensor.t())[0]
            s_broad=torch.mm(v_broad,d_embs_tensor.t())[0]
            s_tech=torch.mm(v_tech,d_embs_tensor.t())[0]
            s_expl=torch.mm(v_expl,d_embs_tensor.t())[0]
            
            
            d_para=s_para - s_base
            d_broad=s_broad - s_base
            d_tech=s_tech - s_base
            d_expl=s_expl - s_base
            
            q_broadcast=q_base_tensor.expand(d_embs_tensor.size(0), -1)
            final_scores, dynamic_weights = self.router(
                q_broadcast, s_base, d_para, d_broad, d_tech, d_expl
            )
            sorted_scores, sorted_indices = torch.sort(final_scores, descending=True)
            best_scores, best_idxs = sorted_scores[:10], sorted_indices[:10]
            # best_score = float(best_scores[0].item())
            # best_doc = candidate_texts[best_idxs[0]]
            # best_weights = dynamic_weights[best_idxs[0]].tolist()
            # print(f"Selected Doc (Index {best_idxs[0]}) | Score: {best_score:.4f}")
            # print(f"Active Intent Weights -> Para:{best_weights[0]:.2f} Broad:{best_weights[1]:.2f} Tech:{best_weights[2]:.2f} Expl:{best_weights[3]:.2f}")
            # print("best doc:", best_doc)
            selected_texts = []
            total_chars = 0
            for idx in best_idxs.tolist()[:MAX_CONTEXT_DOCS]:
                text = candidate_texts[idx]
                if not isinstance(text, str):
                    text = str(text)
                text = text[:MAX_CONTEXT_DOC_CHARS]

                # Enforce total size cap
                if total_chars + len(text) > MAX_CONTEXT_TOTAL_CHARS:
                    remaining = MAX_CONTEXT_TOTAL_CHARS - total_chars
                    if remaining <= 0:
                        break
                    text = text[:remaining]

                selected_texts.append(text)
                total_chars += len(text)
                if total_chars >= MAX_CONTEXT_TOTAL_CHARS:
                    break

            context = "\n--------------------------\n".join(selected_texts)
            # print(context)
            return {"response_docs": context, "scores": best_scores.tolist()}
    def add_to_knowledge_base(self, documents):
        if not self.vectorstore:
            print("Vectorstore not initialized. Cannot add documents.")
            return
        self.vectorstore.add_documents(documents)
        self.vectorstore.save_local("artifacts/vector_db/agri_faiss_index")
        print(f"Added {len(documents)} documents to the knowledge base.")
rag_engine = RAGEngine()
# import asyncio

# rag_engine = RAGEngine()

# async def main():
#     result = await rag_engine.process("Red bugs in my flour")
#     print(result)

# asyncio.run(main())

async def get_rag_engine():
    return rag_engine