"""CLI for tri-LLM hallucination benchmark.

Example:
python examples/tri_llm_benchmark.py \
  --openai-model gpt-4o-mini \
  --ollama-model-a llama3.2:latest \
  --ollama-model-b mistral:latest \
  --prompts-file examples/prompts.txt \
  --num-prompts 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hallucination_detector.tri_llm.config import TriLLMConfig, TriLLMThresholds
from hallucination_detector.tri_llm.pipeline import TriLLMBenchmark


DEFAULT_PROMPTS = [
    "Who won the Nobel Prize in Physics in 2024 and for what contribution?",
    "List five verified causes of climate change and cite at least one recent IPCC fact.",
    "Summarize latest confirmed NASA Mars rover findings with dates.",
    "Explain differences between Type 1 and Type 2 diabetes in 6 bullet points.",
    "Name three major cybersecurity incidents in 2023 and their validated impacts.",
]


def load_prompts(prompts_file: str | None, prompt_json: str | None, num_prompts: int) -> List[str]:
    """Load prompts from file or JSON string, fallback to defaults."""
    prompts: List[str]
    if prompts_file:
        with open(prompts_file, "r", encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle.readlines() if line.strip()]
    elif prompt_json:
        prompts = [item.strip() for item in json.loads(prompt_json) if str(item).strip()]
    else:
        prompts = DEFAULT_PROMPTS

    if num_prompts > 0:
        prompts = prompts[:num_prompts]
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tri-LLM hallucination detection benchmark")

    parser.add_argument("--openai-model", required=True, help="OpenAI model name")
    parser.add_argument("--ollama-model-a", required=True, help="Local Ollama model A")
    parser.add_argument("--ollama-model-b", required=True, help="Local Ollama model B")

    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--openai-base-url", default="https://api.openai.com/v1", help="OpenAI API base URL")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434", help="Ollama base URL")

    parser.add_argument("--prompts-file", default=None, help="Path to newline-separated prompts")
    parser.add_argument("--prompts-json", default=None, help="JSON list of prompts")
    parser.add_argument("--num-prompts", type=int, default=0, help="Use only first N prompts")

    parser.add_argument("--request-timeout", type=int, default=90, help="Per-request timeout in seconds")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max generated tokens")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Sentence-transformers model")

    parser.add_argument("--medium-risk-threshold", type=float, default=40.0)
    parser.add_argument("--high-risk-threshold", type=float, default=70.0)
    parser.add_argument("--severe-divergence-threshold", type=float, default=0.45)
    parser.add_argument("--overconfidence-threshold", type=float, default=55.0)

    parser.add_argument("--output", default="tri_llm_benchmark_results.json", help="Output JSON file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts_file, args.prompts_json, args.num_prompts)
    if not prompts:
        raise ValueError("No prompts supplied. Provide --prompts-file or --prompts-json.")

    thresholds = TriLLMThresholds(
        high_risk_score=args.high_risk_threshold,
        medium_risk_score=args.medium_risk_threshold,
        severe_divergence_threshold=args.severe_divergence_threshold,
        overconfidence_marker_threshold=args.overconfidence_threshold,
    )

    config = TriLLMConfig(
        openai_model=args.openai_model,
        ollama_model_a=args.ollama_model_a,
        ollama_model_b=args.ollama_model_b,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        ollama_base_url=args.ollama_base_url,
        prompts=prompts,
        request_timeout_seconds=args.request_timeout,
        generation_temperature=args.temperature,
        generation_max_tokens=args.max_tokens,
        embedding_model_name=args.embedding_model,
        thresholds=thresholds,
        output_path=args.output,
        log_level=args.log_level,
    )

    benchmark = TriLLMBenchmark(config)
    report = benchmark.run()
    summary = report["summary"]

    print("\n=== TRI-LLM BENCHMARK SUMMARY ===")
    print(f"Most reliable model: {summary['most_reliable_model']}")
    for model_key, metrics in summary["per_model"].items():
        print(
            f"- {model_key}: consistency={metrics['avg_consistency_score']}, "
            f"hallucinations={metrics['hallucination_frequency']}, "
            f"overconfidence={metrics['overconfidence_frequency']}"
        )
    print(f"Detailed report written to: {args.output}")


if __name__ == "__main__":
    main()
