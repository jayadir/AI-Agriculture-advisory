import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import os
from typing import List, Tuple

class LIGNRanker(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super(LIGNRanker, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.fusion_dim = 768 * 4
        
        # Matches the Training Architecture exactly
        self.gate = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.LayerNorm(512), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid() # Added for 0-1 probability interpretation
        )

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, q_ids, q_mask, doc_ids, doc_mask):
        # 1. Get BERT outputs
        q_out = self.bert(q_ids, attention_mask=q_mask)
        u = self.mean_pooling(q_out, q_mask)
        
        d_out = self.bert(doc_ids, attention_mask=doc_mask)
        v = self.mean_pooling(d_out, doc_mask)
        
        # 2. Normalize (Crucial: Matches training step)
        u = F.normalize(u, p=2, dim=1)
        v = F.normalize(v, p=2, dim=1)
        
        # 3. Interaction Features
        diff = torch.abs(u - v)
        prod = u * v
        fusion = torch.cat((u, v, diff, prod), dim=1)
        
        return self.gate(fusion).squeeze(-1)

class LIGNReranker:
    def __init__(self, model_path: str = "artifacts/models/lign_best_model.pth", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = LIGNRanker().to(self.device)
        self.model_path = model_path
        if os.path.exists(model_path):
            # Load weights; strict=False is safer if there are minor metadata mismatches
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            print(f"[LIGN] Model loaded successfully from {model_path}")
        else:
            print(f"[LIGN] Warning: {model_path} not found.")

    def score(self, query: str, document: str) -> float:
        with torch.no_grad():
            q_input = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
            d_input = self.tokenizer(document, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
            
            score = self.model(
                q_input['input_ids'], q_input['attention_mask'],
                d_input['input_ids'], d_input['attention_mask']
            )
        return score.item()

    def batch_score(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        if not documents: return []
        
        with torch.no_grad():
            # Prepare Batch
            q_input = self.tokenizer([query] * len(documents), return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
            d_input = self.tokenizer(documents, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
            
            scores = self.model(
                q_input['input_ids'], q_input['attention_mask'],
                d_input['input_ids'], d_input['attention_mask']
            ).cpu().tolist()
        
        results = list(zip(documents, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[str]:
        scored_docs = self.batch_score(query, documents)
        if top_k:
            scored_docs = scored_docs[:top_k]
        return [doc for doc, score in scored_docs]

_lign_ranker_instance = None

def get_lign_ranker() -> LIGNReranker:
    global _lign_ranker_instance
    if _lign_ranker_instance is None:
        _lign_ranker_instance = LIGNReranker()
    return _lign_ranker_instance