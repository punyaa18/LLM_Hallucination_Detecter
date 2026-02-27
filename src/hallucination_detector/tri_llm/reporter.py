"""Reporting layer for tri-LLM benchmark outputs."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, List

from .config import PromptEvaluation, TriLLMConfig, utc_timestamp


class TriLLMReporter:
    """Builds prompt-level and final benchmark report artifacts."""

    def __init__(self, config: TriLLMConfig):
        self.config = config

    def build(self, prompt_evaluations: List[PromptEvaluation]) -> Dict[str, object]:
        """Build complete report payload for file output."""
        prompt_rows: List[Dict[str, object]] = [self._prompt_to_dict(item) for item in prompt_evaluations]
        summary = self._summary(prompt_evaluations)
        conclusions = self._conclusion(summary)

        return {
            "metadata": {
                "generated_at": utc_timestamp(),
                "openai_model": self.config.openai_model,
                "ollama_model_a": self.config.ollama_model_a,
                "ollama_model_b": self.config.ollama_model_b,
                "num_prompts": len(self.config.prompts),
                "thresholds": asdict(self.config.thresholds),
            },
            "prompt_results": prompt_rows,
            "summary": summary,
            "analytical_conclusion": conclusions,
        }

    def save(self, report_payload: Dict[str, object]) -> None:
        """Persist report to configured output file."""
        with open(self.config.output_path, "w", encoding="utf-8") as handle:
            json.dump(report_payload, handle, indent=2, ensure_ascii=False)

    @staticmethod
    def _prompt_to_dict(item: PromptEvaluation) -> Dict[str, object]:
        return {
            "prompt_index": item.prompt_index,
            "prompt": item.prompt,
            "responses": {
                key: {
                    "provider": value.provider,
                    "model_name": value.model_name,
                    "response": value.response,
                    "latency_seconds": value.latency_seconds,
                    "error": value.error,
                }
                for key, value in item.responses.items()
            },
            "pairwise_similarity": [asdict(pair) for pair in item.pairwise_similarity],
            "per_model": {
                model_key: {
                    "consistency_score": model_eval.consistency_score,
                    "hallucination_risk": model_eval.hallucination_risk,
                    "overconfidence_score": model_eval.overconfidence_score,
                    "estimated_incorrectness_level": model_eval.estimated_incorrectness_level,
                    "divergence_flags": model_eval.divergence_flags,
                    "evidence": model_eval.evidence,
                }
                for model_key, model_eval in item.per_model.items()
            },
        }

    def _summary(self, items: List[PromptEvaluation]) -> Dict[str, object]:
        model_keys = ["openai", "ollama_a", "ollama_b"]
        aggregate: Dict[str, Dict[str, float]] = {
            key: {
                "consistency_sum": 0.0,
                "overconfidence_sum": 0.0,
                "high_risk_count": 0.0,
                "medium_risk_count": 0.0,
                "hallucination_events": 0.0,
                "overconfidence_events": 0.0,
            }
            for key in model_keys
        }

        for prompt_eval in items:
            for key in model_keys:
                entry = prompt_eval.per_model[key]
                bucket = aggregate[key]
                bucket["consistency_sum"] += entry.consistency_score
                bucket["overconfidence_sum"] += entry.overconfidence_score

                if entry.hallucination_risk == "High":
                    bucket["high_risk_count"] += 1
                    bucket["hallucination_events"] += 1
                elif entry.hallucination_risk == "Medium":
                    bucket["medium_risk_count"] += 1
                    bucket["hallucination_events"] += 1

                if entry.overconfidence_score >= self.config.thresholds.overconfidence_marker_threshold:
                    bucket["overconfidence_events"] += 1

        total = max(len(items), 1)
        per_model_summary: Dict[str, Dict[str, float]] = {}
        for key in model_keys:
            row = aggregate[key]
            per_model_summary[key] = {
                "avg_consistency_score": round(row["consistency_sum"] / total, 2),
                "avg_overconfidence_score": round(row["overconfidence_sum"] / total, 2),
                "high_risk_frequency": int(row["high_risk_count"]),
                "medium_risk_frequency": int(row["medium_risk_count"]),
                "hallucination_frequency": int(row["hallucination_events"]),
                "overconfidence_frequency": int(row["overconfidence_events"]),
            }

        reliability_ranked = sorted(
            model_keys,
            key=lambda model: (
                -per_model_summary[model]["avg_consistency_score"],
                per_model_summary[model]["hallucination_frequency"],
                per_model_summary[model]["overconfidence_frequency"],
            ),
        )

        return {
            "per_model": per_model_summary,
            "most_reliable_model": reliability_ranked[0],
            "ranking": reliability_ranked,
        }

    def _conclusion(self, summary: Dict[str, object]) -> Dict[str, object]:
        per_model = summary["per_model"]
        winner = summary["most_reliable_model"]

        key_risks = []
        for key, metrics in per_model.items():
            key_risks.append(
                {
                    "model": key,
                    "hallucination_frequency": metrics["hallucination_frequency"],
                    "overconfidence_frequency": metrics["overconfidence_frequency"],
                }
            )

        return {
            "most_reliable_model": winner,
            "model_risk_profile": key_risks,
            "mitigation_strategies": [
                "Add retrieval-augmented grounding (RAG) with source citation checks.",
                "Apply calibrated refusal policy for low-evidence questions (explicit 'I don't know').",
                "Use confidence scoring layer with threshold-based answer gating.",
                "Run cross-model consensus validation before final answer release.",
                "Continuously evaluate with adversarial factual prompts and update thresholds.",
            ],
        }
