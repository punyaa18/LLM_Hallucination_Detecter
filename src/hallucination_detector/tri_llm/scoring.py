"""Scoring engine for tri-LLM prompt evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .config import TriLLMThresholds
from .evaluator import HeuristicSignals


@dataclass
class ScoreResult:
    """Final model scores for one prompt."""

    consistency_score: float
    hallucination_risk: str
    overconfidence_score: float
    estimated_incorrectness_level: str


class TriLLMScorer:
    """Combines semantic alignment + heuristics into risk and consistency."""

    def __init__(self, thresholds: TriLLMThresholds):
        self.thresholds = thresholds

    def score(self, model_similarity: float, signals: HeuristicSignals, divergence_flags: List[str]) -> ScoreResult:
        """Compute the requested metrics for a single model answer."""
        consistency = max(0.0, min(100.0, model_similarity * 100.0))

        risk_raw = self._risk_numeric(consistency, signals, divergence_flags)
        hallucination_risk = self._risk_bucket(risk_raw)
        incorrectness = self._incorrectness_level(risk_raw)

        return ScoreResult(
            consistency_score=round(consistency, 2),
            hallucination_risk=hallucination_risk,
            overconfidence_score=round(signals.overconfidence_score, 2),
            estimated_incorrectness_level=incorrectness,
        )

    def _risk_numeric(self, consistency: float, signals: HeuristicSignals, divergence_flags: List[str]) -> float:
        inconsistency_component = max(0.0, 100.0 - consistency)
        fabricated_component = signals.fabricated_detail_index * 35.0
        unsupported_component = float(signals.unsupported_specific_claims) * 15.0
        overconfidence_component = signals.overconfidence_score * 0.35
        divergence_component = float(len(divergence_flags)) * 8.0
        idk_component = 12.0 if signals.speculative_without_uncertainty else 0.0

        raw = (
            inconsistency_component * 0.45
            + fabricated_component
            + unsupported_component
            + overconfidence_component
            + divergence_component
            + idk_component
        )
        return max(0.0, min(100.0, raw))

    def _risk_bucket(self, risk_score: float) -> str:
        if risk_score >= self.thresholds.high_risk_score:
            return "High"
        if risk_score >= self.thresholds.medium_risk_score:
            return "Medium"
        return "Low"

    @staticmethod
    def _incorrectness_level(risk_score: float) -> str:
        if risk_score >= 75:
            return "Likely Incorrect"
        if risk_score >= 45:
            return "Possibly Incorrect"
        return "Likely Correct"
