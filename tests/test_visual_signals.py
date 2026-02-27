"""Tests for visual hallucination warning signals."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hallucination_detector.data_models import (
    Claim,
    ClaimVerification,
    HallucinationReport,
    RiskLevel,
    VerificationStatus,
)
from hallucination_detector.visual_signals import compute_warning_level, report_to_visual_payload


def _verification(text: str, status: VerificationStatus, confidence: float) -> ClaimVerification:
    return ClaimVerification(
        claim=Claim(text=text, start_idx=0, end_idx=len(text), confidence=0.9),
        status=status,
        confidence_score=confidence,
        explanation=f"status={status.value}",
    )


def test_compute_warning_level_safe():
    report = HallucinationReport(
        original_text="safe text",
        claims=[Claim(text="Earth orbits Sun", start_idx=0, end_idx=16)],
        verifications=[_verification("Earth orbits Sun", VerificationStatus.SUPPORTED, 0.95)],
        hallucination_score=0.1,
        risk_level=RiskLevel.LOW,
    )
    assert compute_warning_level(report) == "safe"


def test_compute_warning_level_watch():
    report = HallucinationReport(
        original_text="watch text",
        claims=[Claim(text="Claim A", start_idx=0, end_idx=7)],
        verifications=[
            _verification("Claim A", VerificationStatus.UNVERIFIABLE, 0.51),
            _verification("Claim B", VerificationStatus.SUPPORTED, 0.77),
        ],
        hallucination_score=0.4,
        risk_level=RiskLevel.MEDIUM,
    )
    assert compute_warning_level(report) == "watch"


def test_compute_warning_level_danger():
    report = HallucinationReport(
        original_text="danger text",
        claims=[Claim(text="Claim A", start_idx=0, end_idx=7)],
        verifications=[
            _verification("Claim A", VerificationStatus.CONTRADICTED, 0.88),
            _verification("Claim B", VerificationStatus.UNVERIFIABLE, 0.54),
        ],
        hallucination_score=0.8,
        risk_level=RiskLevel.HIGH,
    )
    assert compute_warning_level(report) == "danger"


def test_report_to_visual_payload_shape():
    report = HallucinationReport(
        original_text="test",
        claims=[Claim(text="x", start_idx=0, end_idx=1)],
        verifications=[_verification("x", VerificationStatus.SUPPORTED, 0.9)],
        hallucination_score=0.2,
        risk_level=RiskLevel.LOW,
    )

    payload = report_to_visual_payload(report)

    assert payload["risk_level"] == "low"
    assert "warning_level" in payload
    assert isinstance(payload["claims"], list)
    assert payload["claims"][0]["status"] == "supported"
