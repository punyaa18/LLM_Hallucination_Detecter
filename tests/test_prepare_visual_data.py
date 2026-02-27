"""Tests for visual dataset preparation script."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from examples.prepare_visual_data import prepare_seed_dataset, SEED_RECORDS


def test_prepare_seed_dataset_writes_jsonl(tmp_path):
    output = tmp_path / "visual_seed_dataset.jsonl"
    row_count = prepare_seed_dataset(output)

    assert row_count == len(SEED_RECORDS)
    assert output.exists()

    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(SEED_RECORDS)

    first_row = json.loads(lines[0])
    assert "id" in first_row
    assert "text" in first_row
    assert "expected_warning" in first_row


def test_seed_records_have_expected_labels():
    valid_labels = {"safe", "watch", "danger"}
    for record in SEED_RECORDS:
        assert record["expected_warning"] in valid_labels
