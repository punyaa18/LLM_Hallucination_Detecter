"""Visual signaling helpers for hallucination risk UX."""

from typing import Dict, List

from .data_models import HallucinationReport, VerificationStatus


def compute_warning_level(report: HallucinationReport) -> str:
    """Compute a simple warning level for "about to hallucinate" visuals.

    Levels:
    - safe: low probability of hallucination
    - watch: elevated risk / potentially about to hallucinate
    - danger: high hallucination risk
    """
    total_claims = len(report.verifications)
    problematic = sum(
        1
        for verification in report.verifications
        if verification.status in (VerificationStatus.CONTRADICTED, VerificationStatus.UNVERIFIABLE)
    )
    problematic_ratio = (problematic / total_claims) if total_claims else 0.0

    if report.hallucination_score >= 0.65 or problematic_ratio >= 0.6:
        return "danger"
    if report.hallucination_score >= 0.35 or problematic_ratio >= 0.3:
        return "watch"
    return "safe"


def warning_message(warning_level: str) -> str:
    """Return user-facing message for warning level."""
    if warning_level == "danger":
        return "High risk: local LLM is likely hallucinating now."
    if warning_level == "watch":
        return "Caution: local LLM appears close to hallucination behavior."
    return "Stable: no strong hallucination signal detected yet."


def report_to_visual_payload(report: HallucinationReport) -> Dict[str, object]:
    """Transform report into a UI-friendly JSON payload."""
    summary = report.get_hallucination_summary()
    warning_level = compute_warning_level(report)

    claim_rows: List[Dict[str, object]] = []
    for verification in report.verifications:
        claim_rows.append(
            {
                "claim": verification.claim.text,
                "status": verification.status.value,
                "confidence": round(verification.confidence_score, 3),
                "explanation": verification.explanation,
            }
        )

    return {
        "hallucination_score": round(report.hallucination_score, 3),
        "risk_level": report.risk_level.value,
        "warning_level": warning_level,
        "warning_message": warning_message(warning_level),
        "summary": summary,
        "total_claims": len(report.claims),
        "claims": claim_rows,
    }
