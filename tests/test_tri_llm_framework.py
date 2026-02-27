"""Unit tests for tri-LLM hallucination framework."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hallucination_detector.tri_llm.config import (
    ModelResponse,
    PairSimilarity,
    PromptEvaluation,
    PromptModelEvaluation,
    TriLLMConfig,
    TriLLMThresholds,
)
from hallucination_detector.tri_llm.evaluator import HeuristicSignals, ResponseEvaluator
from hallucination_detector.tri_llm.reporter import TriLLMReporter
from hallucination_detector.tri_llm.scoring import TriLLMScorer


def test_evaluator_detects_overconfidence_and_speculation():
    text = (
        "This is definitely the exact answer and absolutely guaranteed. "
        "In 2024 there were 12 events, 9 treaties, 7 summits, 6 protocols and 5 disputes signed by "
        "Arkania, Belvaria, Novara, Trellon, Caldor, Vensar, Mirath, Solvia and Brenton. "
        "They always happen in Novara and Trellon."
    )
    signals = ResponseEvaluator.evaluate_signals(text)

    assert signals.overconfidence_score > 50
    assert signals.fabricated_detail_index > 0
    assert signals.unsupported_specific_claims >= 1


def test_scorer_maps_high_risk_and_incorrectness():
    thresholds = TriLLMThresholds(high_risk_score=70, medium_risk_score=40)
    scorer = TriLLMScorer(thresholds)
    signals = HeuristicSignals(
        overconfidence_score=80,
        unsupported_specific_claims=2,
        fabricated_detail_index=0.8,
        speculative_without_uncertainty=True,
        contradiction_like_phrases=2,
    )

    result = scorer.score(model_similarity=0.2, signals=signals, divergence_flags=["factual_divergence", "fabricated_details"])

    assert result.hallucination_risk == "High"
    assert result.estimated_incorrectness_level in {"Possibly Incorrect", "Likely Incorrect"}


def test_reporter_ranking_and_frequencies():
    config = TriLLMConfig(
        openai_model="gpt-4o-mini",
        ollama_model_a="llama3.2:latest",
        ollama_model_b="mistral:latest",
        prompts=["p1"],
    )
    reporter = TriLLMReporter(config)

    prompt_eval = PromptEvaluation(
        prompt_index=1,
        prompt="p1",
        responses={
            "openai": ModelResponse("openai", "openai", "gpt-4o-mini", "p1", "r", 0.3),
            "ollama_a": ModelResponse("ollama_a", "ollama", "llama3.2:latest", "p1", "r", 0.4),
            "ollama_b": ModelResponse("ollama_b", "ollama", "mistral:latest", "p1", "r", 0.5),
        },
        pairwise_similarity=[
            PairSimilarity("openai", "ollama_a", 0.9),
            PairSimilarity("openai", "ollama_b", 0.85),
            PairSimilarity("ollama_a", "ollama_b", 0.8),
        ],
        per_model={
            "openai": PromptModelEvaluation("openai", 92, "Low", 25, "Likely Correct", [], {}),
            "ollama_a": PromptModelEvaluation("ollama_a", 70, "Medium", 60, "Possibly Incorrect", ["factual_divergence"], {}),
            "ollama_b": PromptModelEvaluation("ollama_b", 50, "High", 75, "Likely Incorrect", ["fabricated_details"], {}),
        },
    )

    payload = reporter.build([prompt_eval])
    summary = payload["summary"]

    assert summary["most_reliable_model"] == "openai"
    assert summary["per_model"]["ollama_b"]["hallucination_frequency"] == 1
    assert "mitigation_strategies" in payload["analytical_conclusion"]
