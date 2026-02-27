"""Main orchestration pipeline for tri-LLM hallucination benchmark."""

from __future__ import annotations

import logging
from typing import Dict, List

from .config import PromptEvaluation, PromptModelEvaluation, TriLLMConfig
from .evaluator import ResponseEvaluator
from .reporter import TriLLMReporter
from .runners import TriModelRunner
from .scoring import TriLLMScorer


class TriLLMBenchmark:
    """Runs complete prompt benchmark across OpenAI + two local Ollama models."""

    def __init__(self, config: TriLLMConfig):
        self.config = config
        self._configure_logging()
        self.logger = logging.getLogger(__name__)

        self.runners = TriModelRunner(config)
        self.evaluator = ResponseEvaluator(embedding_model_name=config.embedding_model_name)
        self.scorer = TriLLMScorer(config.thresholds)
        self.reporter = TriLLMReporter(config)

    def run(self) -> Dict[str, object]:
        """Execute full benchmark and return report payload."""
        prompt_evaluations: List[PromptEvaluation] = []

        for index, prompt in enumerate(self.config.prompts, start=1):
            self.logger.info("Running prompt %s/%s", index, len(self.config.prompts))
            responses = self.runners.run_prompt(prompt)
            pairs = self.evaluator.pairwise_similarity(responses)

            per_model: Dict[str, PromptModelEvaluation] = {}
            for model_key, model_response in responses.items():
                signals = self.evaluator.evaluate_signals(model_response.response)
                similarity = self.evaluator.model_similarity_average(model_key, pairs)
                flags = self.evaluator.detect_divergence_flags(
                    model_key=model_key,
                    model_similarity=similarity,
                    pairs=pairs,
                    signals=signals,
                    severe_divergence_threshold=self.config.thresholds.severe_divergence_threshold,
                )
                scores = self.scorer.score(similarity, signals, flags)

                per_model[model_key] = PromptModelEvaluation(
                    model_key=model_key,
                    consistency_score=scores.consistency_score,
                    hallucination_risk=scores.hallucination_risk,
                    overconfidence_score=scores.overconfidence_score,
                    estimated_incorrectness_level=scores.estimated_incorrectness_level,
                    divergence_flags=flags,
                    evidence={
                        "avg_similarity": round(similarity, 3),
                        "unsupported_specific_claims": signals.unsupported_specific_claims,
                        "fabricated_detail_index": signals.fabricated_detail_index,
                        "speculative_without_uncertainty": signals.speculative_without_uncertainty,
                        "raw_overconfidence": signals.overconfidence_score,
                        "error": model_response.error,
                    },
                )

            prompt_evaluations.append(
                PromptEvaluation(
                    prompt_index=index,
                    prompt=prompt,
                    responses=responses,
                    pairwise_similarity=pairs,
                    per_model=per_model,
                )
            )

        report_payload = self.reporter.build(prompt_evaluations)
        self.reporter.save(report_payload)
        self.logger.info("Saved report to %s", self.config.output_path)
        return report_payload

    def _configure_logging(self) -> None:
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=getattr(logging, self.config.log_level.upper(), logging.INFO),
                format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            )
