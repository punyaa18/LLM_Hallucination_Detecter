"""Benchmark hallucination behavior across local Ollama models and optional online model.

Usage examples:
  python examples/compare_model_hallucinations.py --model-a llama3.2:latest --model-b mistral:latest
  python examples/compare_model_hallucinations.py --model-a llama3.2:latest --model-b mistral:latest --online-provider openrouter --online-model openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from config.config import DetectorConfig
from hallucination_detector.detector import HallucinationDetector
from hallucination_detector.utils import get_statistics


DEFAULT_PROMPTS = [
    "Name five scientifically accepted causes of climate change and cite one specific IPCC finding.",
    "What are the key contributions of Alan Turing to computer science?",
    "Explain the difference between Type 1 and Type 2 diabetes in 6 bullet points.",
    "Summarize the latest confirmed Mars rover discoveries from NASA with dates.",
    "Who won the Nobel Prize in Physics in 2024 and why?",
    "Describe how mRNA vaccines work and list two known limitations.",
    "List three major cybersecurity incidents in 2023 with verified impacts.",
    "Give a concise timeline of CRISPR milestones from 2012 onward.",
]


@dataclass
class PromptResult:
    prompt: str
    response: str
    latency_seconds: float
    hallucination_score: float
    risk_level: str
    supported_claims: int
    contradicted_claims: int
    unverifiable_claims: int
    total_claims: int


def call_ollama(
    base_url: str,
    model: str,
    prompt: str,
    timeout: int,
    temperature: float,
    num_predict: int,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "").strip()


def call_openrouter(
    model: str,
    prompt: str,
    api_key: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def call_openai(
    model: str,
    prompt: str,
    api_key: str,
    timeout: int,
    temperature: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_response(
    runner_type: str,
    model: str,
    prompt: str,
    args: argparse.Namespace,
) -> str:
    if runner_type == "ollama":
        return call_ollama(
            base_url=args.ollama_base_url,
            model=model,
            prompt=prompt,
            timeout=args.request_timeout,
            temperature=args.generation_temperature,
            num_predict=args.generation_max_tokens,
        )

    if runner_type == "openrouter":
        if not args.online_api_key:
            raise ValueError("Missing online API key. Set --online-api-key or OPENROUTER_API_KEY.")
        return call_openrouter(
            model=model,
            prompt=prompt,
            api_key=args.online_api_key,
            timeout=args.request_timeout,
            temperature=args.generation_temperature,
            max_tokens=args.generation_max_tokens,
        )

    if runner_type == "openai":
        if not args.online_api_key:
            raise ValueError("Missing online API key. Set --online-api-key or OPENAI_API_KEY.")
        return call_openai(
            model=model,
            prompt=prompt,
            api_key=args.online_api_key,
            timeout=args.request_timeout,
            temperature=args.generation_temperature,
            max_tokens=args.generation_max_tokens,
        )

    raise ValueError(f"Unsupported runner type: {runner_type}")


def evaluate_model(
    detector: HallucinationDetector,
    runner_type: str,
    model: str,
    prompts: List[str],
    args: argparse.Namespace,
) -> List[PromptResult]:
    results: List[PromptResult] = []

    for idx, prompt in enumerate(prompts, start=1):
        print(f"[{model}] Prompt {idx}/{len(prompts)}")
        started = time.perf_counter()
        response_text = generate_response(runner_type, model, prompt, args)
        latency = time.perf_counter() - started

        report = detector.detect(response_text)
        stats = get_statistics(report)

        result = PromptResult(
            prompt=prompt,
            response=response_text,
            latency_seconds=round(latency, 3),
            hallucination_score=float(stats["hallucination_score"]),
            risk_level=stats["risk_level"],
            supported_claims=int(stats["supported"]),
            contradicted_claims=int(stats["contradicted"]),
            unverifiable_claims=int(stats["unverifiable"]),
            total_claims=int(stats["total_claims"]),
        )
        results.append(result)

    return results


def aggregate_results(results: List[PromptResult], threshold: float) -> Dict[str, object]:
    scores = [r.hallucination_score for r in results]
    latencies = [r.latency_seconds for r in results]
    flagged = [r for r in results if r.hallucination_score >= threshold]

    first_hallucination_prompt: Optional[int] = None
    for index, item in enumerate(results, start=1):
        if item.hallucination_score >= threshold:
            first_hallucination_prompt = index
            break

    return {
        "avg_hallucination_score": round(statistics.mean(scores), 4) if scores else 0.0,
        "median_hallucination_score": round(statistics.median(scores), 4) if scores else 0.0,
        "avg_latency_seconds": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "hallucination_rate": round(len(flagged) / len(results), 3) if results else 0.0,
        "first_hallucination_prompt_index": first_hallucination_prompt,
        "total_supported_claims": sum(r.supported_claims for r in results),
        "total_contradicted_claims": sum(r.contradicted_claims for r in results),
        "total_unverifiable_claims": sum(r.unverifiable_claims for r in results),
    }


def compare_models(summary: Dict[str, Dict[str, object]], threshold: float) -> Dict[str, object]:
    names = list(summary.keys())
    ranked = sorted(
        names,
        key=lambda name: (
            summary[name]["avg_hallucination_score"],
            summary[name]["hallucination_rate"],
            summary[name]["avg_latency_seconds"],
        ),
    )

    best = ranked[0]
    notes: List[str] = [
        f"Best reliability by this benchmark: {best}.",
        f"Hallucination threshold used: {threshold:.2f} (lower is stricter).",
    ]

    for name in ranked:
        item = summary[name]
        notes.append(
            f"{name}: avg_score={item['avg_hallucination_score']}, rate={item['hallucination_rate']}, avg_latency={item['avg_latency_seconds']}s"
        )

    local_models = [name for name in names if not name.startswith("online:")]
    online_models = [name for name in names if name.startswith("online:")]

    if local_models:
        notes.append(
            "Why local models are often preferred: data stays on-device, predictable cost, and offline availability."
        )
    if online_models:
        notes.append(
            "Why online models are often preferred: broader hosted compute/options and no local model management."
        )

    return {
        "ranked_models": ranked,
        "best_model": best,
        "notes": notes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare hallucination performance of two local Ollama models and optional online baseline."
    )
    parser.add_argument("--model-a", required=True, help="First local Ollama model name")
    parser.add_argument("--model-b", required=True, help="Second local Ollama model name")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434", help="Ollama server base URL")

    parser.add_argument(
        "--online-provider",
        choices=["none", "openrouter", "openai"],
        default="none",
        help="Optional online provider for third-model baseline",
    )
    parser.add_argument("--online-model", default="", help="Online model ID (required if online provider != none)")
    parser.add_argument(
        "--online-api-key",
        default="",
        help="API key for online provider (or use OPENROUTER_API_KEY / OPENAI_API_KEY)",
    )

    parser.add_argument(
        "--hallucination-threshold",
        type=float,
        default=0.35,
        help="Prompt considered hallucinated when score >= threshold",
    )
    parser.add_argument(
        "--detector-nli-backend",
        choices=["hf", "ollama"],
        default="hf",
        help="Backend used for NLI verification inside detector",
    )
    parser.add_argument(
        "--detector-embedding-backend",
        choices=["hf", "ollama"],
        default="hf",
        help="Backend used for evidence embeddings inside detector",
    )
    parser.add_argument("--max-prompts", type=int, default=8, help="Number of prompts to run from built-in set")
    parser.add_argument("--request-timeout", type=int, default=90, help="HTTP timeout per generation request")
    parser.add_argument("--generation-temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--generation-max-tokens", type=int, default=256, help="Max generated tokens")
    parser.add_argument(
        "--output-json",
        default="benchmark_results.json",
        help="Output path for full JSON report",
    )
    return parser.parse_args()


def resolve_online_key(args: argparse.Namespace) -> str:
    if args.online_api_key:
        return args.online_api_key
    if args.online_provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY", "")
    if args.online_provider == "openai":
        return os.getenv("OPENAI_API_KEY", "")
    return ""


def main() -> None:
    args = parse_args()

    if args.online_provider != "none" and not args.online_model:
        raise ValueError("--online-model is required when --online-provider is set.")

    args.online_api_key = resolve_online_key(args)

    prompts = DEFAULT_PROMPTS[: max(args.max_prompts, 1)]

    detector_config = DetectorConfig(
        verbose=False,
        nli_backend=args.detector_nli_backend,
        embedding_backend=args.detector_embedding_backend,
        max_claims_per_text=10,
    )
    detector = HallucinationDetector(detector_config)

    model_runs: List[Tuple[str, str, str]] = [
        (f"local:{args.model_a}", "ollama", args.model_a),
        (f"local:{args.model_b}", "ollama", args.model_b),
    ]
    if args.online_provider != "none":
        model_runs.append((f"online:{args.online_model}", args.online_provider, args.online_model))

    per_model_results: Dict[str, List[PromptResult]] = {}
    summary: Dict[str, Dict[str, object]] = {}

    for display_name, runner_type, model_name in model_runs:
        results = evaluate_model(detector, runner_type, model_name, prompts, args)
        per_model_results[display_name] = results
        summary[display_name] = aggregate_results(results, args.hallucination_threshold)

    comparison = compare_models(summary, args.hallucination_threshold)

    output = {
        "config": {
            "model_a": args.model_a,
            "model_b": args.model_b,
            "online_provider": args.online_provider,
            "online_model": args.online_model,
            "hallucination_threshold": args.hallucination_threshold,
            "prompt_count": len(prompts),
        },
        "summary": summary,
        "comparison": comparison,
        "results": {
            model_name: [
                {
                    "prompt": row.prompt,
                    "response": row.response,
                    "latency_seconds": row.latency_seconds,
                    "hallucination_score": row.hallucination_score,
                    "risk_level": row.risk_level,
                    "supported_claims": row.supported_claims,
                    "contradicted_claims": row.contradicted_claims,
                    "unverifiable_claims": row.unverifiable_claims,
                    "total_claims": row.total_claims,
                }
                for row in rows
            ]
            for model_name, rows in per_model_results.items()
        },
    }

    output_path = PROJECT_ROOT / args.output_json
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n=== Model Comparison Summary ===")
    for model_name, metrics in summary.items():
        print(
            f"{model_name} | avg_score={metrics['avg_hallucination_score']} | "
            f"rate={metrics['hallucination_rate']} | avg_latency={metrics['avg_latency_seconds']}s | "
            f"first_hallucination_prompt={metrics['first_hallucination_prompt_index']}"
        )

    print("\n=== Preference Notes ===")
    for note in comparison["notes"]:
        print(f"- {note}")

    print(f"\nSaved full benchmark report to: {output_path}")


if __name__ == "__main__":
    main()
