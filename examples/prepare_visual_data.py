"""Prepare small dataset for visual hallucination monitoring demos.

Usage:
  python examples/prepare_visual_data.py
  python examples/prepare_visual_data.py --output data/visual_seed_dataset.jsonl
  python examples/prepare_visual_data.py --analyze-live --nli-backend ollama
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from config.config import DetectorConfig
from hallucination_detector.detector import HallucinationDetector
from hallucination_detector.visual_signals import report_to_visual_payload


SEED_RECORDS: List[Dict[str, str]] = [
    {
        "id": "seed-001",
        "text": "Paris is the capital of France. The Eiffel Tower is located in Paris.",
        "expected_warning": "safe",
    },
    {
        "id": "seed-002",
        "text": "Mars has two moons named Phobos and Deimos. Mars has liquid oceans and blue forests.",
        "expected_warning": "watch",
    },
    {
        "id": "seed-003",
        "text": "The Sun is a star. The Sun is made of frozen iron and orbits Earth once per day.",
        "expected_warning": "danger",
    },
    {
        "id": "seed-004",
        "text": "Python was created by Guido van Rossum. Python is a reptile-only programming language.",
        "expected_warning": "watch",
    },
]


def write_jsonl(records: List[Dict[str, object]], output_path: Path) -> int:
    """Write records to JSONL and return number of rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(records)


def prepare_seed_dataset(output_path: Path) -> int:
    """Write static seed dataset for the dashboard."""
    return write_jsonl(SEED_RECORDS, output_path)


def prepare_with_live_analysis(output_path: Path, nli_backend: str) -> int:
    """Run detector over seeds and export visual payloads for each row."""
    config = DetectorConfig(verbose=False, nli_backend=nli_backend)
    detector = HallucinationDetector(config)

    enriched: List[Dict[str, object]] = []
    for row in SEED_RECORDS:
        report = detector.detect(row["text"])
        payload = report_to_visual_payload(report)
        enriched.append(
            {
                "id": row["id"],
                "text": row["text"],
                "expected_warning": row["expected_warning"],
                "detected_warning": payload["warning_level"],
                "hallucination_score": payload["hallucination_score"],
                "risk_level": payload["risk_level"],
                "summary": payload["summary"],
            }
        )

    return write_jsonl(enriched, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare visual hallucination datasets.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "visual_seed_dataset.jsonl"),
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--analyze-live",
        action="store_true",
        help="Run detector and save predicted warning labels (requires model stack).",
    )
    parser.add_argument(
        "--nli-backend",
        default="hf",
        choices=["hf", "ollama"],
        help="NLI backend used when --analyze-live is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    if args.analyze_live:
        rows = prepare_with_live_analysis(output_path, nli_backend=args.nli_backend)
    else:
        rows = prepare_seed_dataset(output_path)

    print(f"Wrote {rows} rows to {output_path}")


if __name__ == "__main__":
    main()
