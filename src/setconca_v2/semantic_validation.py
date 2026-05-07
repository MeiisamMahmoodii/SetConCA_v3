from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class SemanticValidationResult:
    passed: bool
    reasons: List[str]
    metrics: Dict[str, float | None]


class SemanticValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.nli_enabled = bool(self.config.get("nli_enabled", False))
        self.embedding_model = None
        self.nli_pipeline = None

        if self.enabled:
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer(self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))

        if self.enabled and self.nli_enabled:
            from transformers import pipeline

            self.nli_pipeline = pipeline(
                "text-classification",
                model=self.config.get("nli_model", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"),
                top_k=None,
                truncation=True,
            )

    def validate(self, original: str, rewrite: str) -> SemanticValidationResult:
        if not self.enabled:
            return SemanticValidationResult(True, [], {"embedding_cosine": None, "entailment": None, "contradiction": None})

        reasons: List[str] = []
        metrics: Dict[str, float | None] = {"embedding_cosine": None, "entailment": None, "contradiction": None}

        emb = self.embedding_model.encode([original, rewrite], normalize_embeddings=True)
        cosine = float(np.dot(emb[0], emb[1]))
        metrics["embedding_cosine"] = cosine
        min_cos = float(self.config.get("min_embedding_cosine", 0.62))
        if cosine < min_cos:
            reasons.append(f"embedding_cosine={cosine:.3f}<min={min_cos:.3f}")

        if self.nli_pipeline is not None:
            pair = {"text": original, "text_pair": rewrite}
            raw = self.nli_pipeline(pair)
            labels = {}
            for item in raw[0] if raw and isinstance(raw[0], list) else raw:
                labels[item["label"].lower()] = float(item["score"])
            entail = labels.get("entailment", labels.get("entails"))
            contra = labels.get("contradiction", labels.get("contradicts"))
            metrics["entailment"] = entail
            metrics["contradiction"] = contra
            min_entail = float(self.config.get("min_entailment", 0.50))
            max_contra = float(self.config.get("max_contradiction", 0.25))
            if entail is not None and entail < min_entail:
                reasons.append(f"entailment={entail:.3f}<min={min_entail:.3f}")
            if contra is not None and contra > max_contra:
                reasons.append(f"contradiction={contra:.3f}>max={max_contra:.3f}")

        return SemanticValidationResult(not reasons, reasons, metrics)

    def close(self) -> None:
        self.embedding_model = None
        self.nli_pipeline = None

