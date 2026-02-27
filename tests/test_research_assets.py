"""Validation tests for research paper assets (data + figures)."""

from pathlib import Path
import csv
import json


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "Hallucinations in Local Large Language Models"
DATA_DIR = PAPER_DIR / "data"
FIGURES_DIR = PAPER_DIR / "figures"


def test_research_data_files_exist():
    assert (DATA_DIR / "prompt_level_metrics.csv").exists()
    assert (DATA_DIR / "benchmark_summary.json").exists()


def test_prompt_level_metrics_shape_and_columns():
    csv_path = DATA_DIR / "prompt_level_metrics.csv"
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 24
    required = {
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
    }
    assert set(reader.fieldnames or []) == required



def test_summary_and_ranking_consistency():
    summary_path = DATA_DIR / "benchmark_summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))

    assert data["num_prompts"] == 8
    assert len(data["models"]) == 3
    assert data["most_reliable_model"] == data["ranking"][0]

    for model in data["models"]:
        assert model in data["summary"]
        assert 0 <= data["summary"][model]["avg_consistency"] <= 100
        assert 0 <= data["summary"][model]["avg_overconfidence"] <= 100



def test_figure_files_exist_and_nonempty():
    figures = [
        "consistency_scores.png",
        "hallucination_frequency.png",
        "overconfidence_scores.png",
    ]
    for fig in figures:
        path = FIGURES_DIR / fig
        assert path.exists()
        assert path.stat().st_size > 1000
