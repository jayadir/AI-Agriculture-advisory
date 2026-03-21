import gc
import os
from typing import Dict, List

import torch
from sentence_transformers import CrossEncoder

from app.core.config import settings

FORCE_CPU = os.getenv("FORCE_CPU", "0").strip().lower() in ("1", "true", "yes", "y", "on")


class CrossEncoderReranker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CrossEncoderReranker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.available = False
        self.model_name = settings.RERANKER_MODEL_NAME
        self.batch_size = settings.RERANKER_BATCH_SIZE

        if FORCE_CPU:
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"⏳ Loading reranker `{self.model_name}` on {self.device}...")

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            self.model = CrossEncoder(self.model_name, device=self.device)
            self.available = True
            print(f"✅ Reranker loaded successfully: {self.model_name}")
        except Exception as error:
            self.model = None
            print(f"⚠️ Failed to load reranker `{self.model_name}`: {error}")

    def rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        if not candidates:
            return []

        if not self.available or self.model is None:
            fallback = []
            for index, candidate in enumerate(candidates, start=1):
                candidate_copy = dict(candidate)
                candidate_copy["rerank_score"] = 0.0
                candidate_copy["rerank_rank"] = index
                fallback.append(candidate_copy)
            return fallback[:top_k]

        pairs = [(query, candidate["content"]) for candidate in candidates]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        reranked = []
        for candidate, score in zip(candidates, scores):
            candidate_copy = dict(candidate)
            candidate_copy["rerank_score"] = float(score)
            reranked.append(candidate_copy)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)

        for index, candidate in enumerate(reranked, start=1):
            candidate["rerank_rank"] = index

        return reranked[:top_k]


_reranker_instance = None


def get_reranker() -> CrossEncoderReranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker()
    return _reranker_instance