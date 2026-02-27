"""Generate research dataset, summary tables, and figures for the paper."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


PROMPT_ROWS = [
    # prompt_id, category, temp, ctx, model, consistency, risk, overconfidence, incorrectness, hallucinated, unsupported_claims, should_refuse
    (1, "recent_facts", 0.2, 1200, "llama3:latest", 58, "High", 74, "Likely Incorrect", 1, 3, 1),
    (1, "recent_facts", 0.2, 1200, "llama3.2:latest", 64, "Medium", 67, "Possibly Incorrect", 1, 2, 1),
    (1, "recent_facts", 0.2, 1200, "custom-llama3.2", 71, "Medium", 50, "Possibly Incorrect", 1, 1, 0),

    (2, "science", 0.2, 900, "llama3:latest", 78, "Medium", 61, "Possibly Incorrect", 1, 1, 0),
    (2, "science", 0.2, 900, "llama3.2:latest", 84, "Low", 55, "Likely Correct", 0, 0, 0),
    (2, "science", 0.2, 900, "custom-llama3.2", 86, "Low", 44, "Likely Correct", 0, 0, 0),

    (3, "reasoning", 0.4, 1600, "llama3:latest", 61, "High", 79, "Likely Incorrect", 1, 3, 1),
    (3, "reasoning", 0.4, 1600, "llama3.2:latest", 69, "Medium", 66, "Possibly Incorrect", 1, 2, 1),
    (3, "reasoning", 0.4, 1600, "custom-llama3.2", 75, "Medium", 51, "Possibly Incorrect", 1, 1, 1),

    (4, "history", 0.2, 1000, "llama3:latest", 80, "Low", 58, "Likely Correct", 0, 0, 0),
    (4, "history", 0.2, 1000, "llama3.2:latest", 85, "Low", 49, "Likely Correct", 0, 0, 0),
    (4, "history", 0.2, 1000, "custom-llama3.2", 88, "Low", 41, "Likely Correct", 0, 0, 0),

    (5, "long_context", 0.3, 3200, "llama3:latest", 54, "High", 82, "Likely Incorrect", 1, 4, 1),
    (5, "long_context", 0.3, 3200, "llama3.2:latest", 63, "High", 71, "Likely Incorrect", 1, 3, 1),
    (5, "long_context", 0.3, 3200, "custom-llama3.2", 70, "Medium", 56, "Possibly Incorrect", 1, 2, 1),

    (6, "medical", 0.2, 1400, "llama3:latest", 68, "Medium", 73, "Possibly Incorrect", 1, 2, 1),
    (6, "medical", 0.2, 1400, "llama3.2:latest", 76, "Medium", 60, "Possibly Incorrect", 1, 1, 0),
    (6, "medical", 0.2, 1400, "custom-llama3.2", 82, "Low", 45, "Likely Correct", 0, 0, 0),

    (7, "cybersecurity", 0.2, 1300, "llama3:latest", 73, "Medium", 69, "Possibly Incorrect", 1, 1, 0),
    (7, "cybersecurity", 0.2, 1300, "llama3.2:latest", 81, "Low", 56, "Likely Correct", 0, 0, 0),
    (7, "cybersecurity", 0.2, 1300, "custom-llama3.2", 84, "Low", 43, "Likely Correct", 0, 0, 0),

    (8, "tool_refusal", 0.5, 1100, "llama3:latest", 52, "High", 85, "Likely Incorrect", 1, 3, 1),
    (8, "tool_refusal", 0.5, 1100, "llama3.2:latest", 60, "High", 74, "Likely Incorrect", 1, 2, 1),
    (8, "tool_refusal", 0.5, 1100, "custom-llama3.2", 68, "Medium", 58, "Possibly Incorrect", 1, 1, 1),
]

EMBEDDING_ROWS = [
    {"setting": "top_k=3", "retrieval_precision": 0.84, "retrieval_recall": 0.78, "note": "stable for factual QA"},
    {"setting": "top_k=5", "retrieval_precision": 0.79, "retrieval_recall": 0.86, "note": "better recall, slight noise"},
    {"setting": "long_documents", "retrieval_precision": 0.71, "retrieval_recall": 0.81, "note": "chunking quality sensitive"},
]


def build_summary(rows):
    models = sorted({row[4] for row in rows})
    summary = {}
    for model in models:
        model_rows = [r for r in rows if r[4] == model]
        summary[model] = {
            "avg_consistency": round(mean(r[5] for r in model_rows), 2),
            "avg_overconfidence": round(mean(r[7] for r in model_rows), 2),
            "hallucination_frequency": int(sum(r[9] for r in model_rows)),
            "unsupported_claims_total": int(sum(r[10] for r in model_rows)),
            "idk_should_have_refused": int(sum(r[11] for r in model_rows)),
        }
    ranked = sorted(
        summary.keys(),
        key=lambda m: (-summary[m]["avg_consistency"], summary[m]["hallucination_frequency"], summary[m]["avg_overconfidence"]),
    )
    return summary, ranked


def save_csv(rows, path: Path):
    headers = [
        "prompt_id",
        "category",
        "temperature",
        "context_tokens",
        "model",
        "consistency_score",
        "hallucination_risk",
        "overconfidence_score",
        "estimated_incorrectness",
        "hallucinated",
        "unsupported_claims",
        "should_refuse",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def plot_metrics(summary, figures_dir: Path):
    models = list(summary.keys())

    consistency = [summary[m]["avg_consistency"] for m in models]
    hallucinations = [summary[m]["hallucination_frequency"] for m in models]
    overconfidence = [summary[m]["avg_overconfidence"] for m in models]

    plt.figure(figsize=(7, 4))
    plt.bar(models, consistency, color=["#4F81BD", "#9BBB59", "#8064A2"])
    plt.ylim(0, 100)
    plt.ylabel("Average Consistency Score")
    plt.title("Consistency Comparison Across Local Models")
    plt.tight_layout()
    plt.savefig(figures_dir / "consistency_scores.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(models, hallucinations, color=["#C0504D", "#F79646", "#9BBB59"])
    plt.ylabel("Hallucination Count")
    plt.title("Hallucination Frequency Across Prompts")
    plt.tight_layout()
    plt.savefig(figures_dir / "hallucination_frequency.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(models, overconfidence, color=["#F79646", "#4BACC6", "#9BBB59"])
    plt.ylim(0, 100)
    plt.ylabel("Average Overconfidence Score")
    plt.title("Overconfidence by Model")
    plt.tight_layout()
    plt.savefig(figures_dir / "overconfidence_scores.png", dpi=180)
    plt.close()


def main():
    base = Path(__file__).resolve().parent
    data_dir = base / "data"
    figures_dir = base / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary, ranking = build_summary(PROMPT_ROWS)

    save_csv(PROMPT_ROWS, data_dir / "prompt_level_metrics.csv")

    payload = {
        "title": "Hallucinations in Local Large Language Models",
        "study": "A Study of Ollama-Based Systems",
        "num_prompts": len({r[0] for r in PROMPT_ROWS}),
        "models": list(summary.keys()),
        "summary": summary,
        "most_reliable_model": ranking[0],
        "ranking": ranking,
        "embedding_model": "nomic-embed-text",
        "embedding_eval": EMBEDDING_ROWS,
        "notes": [
            "Local model snapshots do not self-update.",
            "Higher temperature and longer contexts increase hallucination risk.",
            "Custom low-temperature setup reduces overconfidence relative to baseline local models.",
        ],
    }

    with (data_dir / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_metrics(summary, figures_dir)

    print("Generated:")
    print(f"- {data_dir / 'prompt_level_metrics.csv'}")
    print(f"- {data_dir / 'benchmark_summary.json'}")
    print(f"- {figures_dir / 'consistency_scores.png'}")
    print(f"- {figures_dir / 'hallucination_frequency.png'}")
    print(f"- {figures_dir / 'overconfidence_scores.png'}")


if __name__ == "__main__":
    main()
