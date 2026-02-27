"""Evaluation logic for tri-LLM divergence and confidence behavior."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import ModelResponse, PairSimilarity


UNCERTAINTY_MARKERS = (
    "i don't know",
    "not sure",
    "uncertain",
    "might",
    "may",
    "possibly",
    "could be",
    "unclear",
)

OVERCONFIDENT_MARKERS = (
    "definitely",
    "certainly",
    "without a doubt",
    "absolutely",
    "always",
    "guaranteed",
)


@dataclass
class HeuristicSignals:
    """Extracted risk signals from one response."""

    overconfidence_score: float
    unsupported_specific_claims: int
    fabricated_detail_index: float
    speculative_without_uncertainty: bool
    contradiction_like_phrases: int


class ResponseEvaluator:
    """Computes semantic and rule-based features for a prompt batch."""

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model_name
        self.encoder = None

    def _get_encoder(self):
        if self.encoder is None:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.embedding_model_name)
        return self.encoder

    def pairwise_similarity(self, responses: Dict[str, ModelResponse]) -> List[PairSimilarity]:
        """Cosine similarity for every response pair."""
        keys = list(responses.keys())
        texts = [responses[key].response or "" for key in keys]
        vectors = self._get_encoder().encode(texts)
        output: List[PairSimilarity] = []

        for i, j in itertools.combinations(range(len(keys)), 2):
            score = self._cosine(vectors[i], vectors[j])
            output.append(PairSimilarity(model_x=keys[i], model_y=keys[j], score=score))
        return output

    @staticmethod
    def evaluate_signals(response_text: str) -> HeuristicSignals:
        """Compute factual-risk and confidence heuristics from a single answer."""
        text = response_text.strip()
        lowered = text.lower()

        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        capitals = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
        has_sources = bool(re.search(r"https?://|according to|source|citation", lowered))

        unsupported_specific_claims = 0
        if (len(numbers) >= 3 or len(capitals) >= 8) and not has_sources:
            unsupported_specific_claims = 1
        if len(numbers) >= 6 and not has_sources:
            unsupported_specific_claims = 2

        overconfident_hits = sum(marker in lowered for marker in OVERCONFIDENT_MARKERS)
        uncertainty_hits = sum(marker in lowered for marker in UNCERTAINTY_MARKERS)

        answer_len = max(len(text.split()), 1)
        overconfidence_score = min(100.0, (overconfident_hits * 25.0) + max(0.0, 20.0 - uncertainty_hits * 8.0))
        if answer_len > 120 and uncertainty_hits == 0:
            overconfidence_score = min(100.0, overconfidence_score + 10.0)

        fabricated_detail_index = min(1.0, (len(numbers) * 0.06) + (max(0, len(capitals) - 5) * 0.03))
        if has_sources:
            fabricated_detail_index = max(0.0, fabricated_detail_index - 0.15)

        speculative_without_uncertainty = (len(text.split()) > 70) and uncertainty_hits == 0

        contradiction_like_phrases = len(re.findall(r"\b(however|but|on the other hand|nevertheless)\b", lowered))

        return HeuristicSignals(
            overconfidence_score=round(overconfidence_score, 2),
            unsupported_specific_claims=unsupported_specific_claims,
            fabricated_detail_index=round(fabricated_detail_index, 3),
            speculative_without_uncertainty=speculative_without_uncertainty,
            contradiction_like_phrases=contradiction_like_phrases,
        )

    @staticmethod
    def _cosine(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def model_similarity_average(self, model_key: str, pairs: List[PairSimilarity]) -> float:
        """Average pairwise semantic alignment for a specific model."""
        scores: List[float] = []
        for pair in pairs:
            if pair.model_x == model_key or pair.model_y == model_key:
                scores.append(pair.score)
        if not scores:
            return 0.0
        return float(sum(scores) / len(scores))

    @staticmethod
    def detect_divergence_flags(
        model_key: str,
        model_similarity: float,
        pairs: List[PairSimilarity],
        signals: HeuristicSignals,
        severe_divergence_threshold: float,
    ) -> List[str]:
        """Generate interpretable divergence/quality flags for reporting."""
        flags: List[str] = []

        if model_similarity < severe_divergence_threshold:
            flags.append("factual_divergence")

        if signals.contradiction_like_phrases >= 2:
            flags.append("possible_internal_contradiction")

        if signals.fabricated_detail_index >= 0.45:
            flags.append("fabricated_details")

        if signals.unsupported_specific_claims > 0:
            flags.append("unsupported_specific_claims")

        if signals.speculative_without_uncertainty:
            flags.append("should_have_said_i_dont_know")

        model_pairs = [pair for pair in pairs if pair.model_x == model_key or pair.model_y == model_key]
        if model_pairs and all(pair.score < severe_divergence_threshold for pair in model_pairs):
            flags.append("cross_model_contradiction")

        return flags
