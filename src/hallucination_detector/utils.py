"""Utility functions for hallucination detection."""

from typing import List, Dict
from .data_models import HallucinationReport, ClaimVerification, VerificationStatus
import json


def highlight_hallucinations(text: str, report: HallucinationReport) -> str:
    """Highlight potentially hallucinated content in the original text.
    
    Args:
        text: Original text.
        report: Hallucination report.
        
    Returns:
        Text with HTML-style highlighting of problematic claims.
    """
    problematic_verifications = report.get_problematic_claims()
    
    if not problematic_verifications:
        return text
    
    # Sort by position (reverse so we can modify without messing indices)
    problematic_verifications.sort(
        key=lambda v: v.claim.start_idx,
        reverse=True
    )
    
    result = text
    for verification in problematic_verifications:
        claim = verification.claim
        status = verification.status
        
        # Create highlight
        if status == VerificationStatus.CONTRADICTED:
            highlight = f"[❌ CONTRADICTED: {claim.text}]"
            color = "red"
        elif status == VerificationStatus.UNVERIFIABLE:
            highlight = f"[⚠️  UNVERIFIABLE: {claim.text}]"
            color = "yellow"
        else:
            continue
        
        # Insert highlight
        result = (
            result[:claim.start_idx] +
            f"<span style='background-color: {color}; text-decoration: underline;'>" +
            claim.text +
            "</span>" +
            result[claim.end_idx:]
        )
    
    return result


def export_report(report: HallucinationReport, filepath: str, format: str = "json") -> None:
    """Export hallucination report to file.
    
    Args:
        report: HallucinationReport to export.
        filepath: Path to save report.
        format: Export format ('json' or 'text').
    """
    if format == "json":
        data = {
            "hallucination_score": round(report.hallucination_score, 3),
            "risk_level": report.risk_level.value,
            "summary": report.summary,
            "original_text": report.original_text,
            "claims": [
                {
                    "text": c.text,
                    "confidence": c.confidence
                }
                for c in report.claims
            ],
            "verifications": [
                {
                    "claim": v.claim.text,
                    "status": v.status.value,
                    "confidence_score": round(v.confidence_score, 3),
                    "explanation": v.explanation,
                    "supporting_evidence": [
                        {
                            "source": e.source,
                            "text": e.text[:200],
                            "url": e.url,
                            "relevance": round(e.relevance_score, 3)
                        }
                        for e in v.supporting_evidence
                    ],
                    "contradicting_evidence": [
                        {
                            "source": e.source,
                            "text": e.text[:200],
                            "url": e.url,
                            "relevance": round(e.relevance_score, 3)
                        }
                        for e in v.contradicting_evidence
                    ]
                }
                for v in report.verifications
            ]
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    
    elif format == "text":
        with open(filepath, "w") as f:
            f.write(f"HALLUCINATION DETECTION REPORT\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Hallucination Score: {report.hallucination_score:.2%}\n")
            f.write(f"Risk Level: {report.risk_level.value.upper()}\n\n")
            f.write(f"Summary: {report.summary}\n\n")
            f.write(f"{'=' * 80}\n\n")
            
            for i, verification in enumerate(report.verifications, 1):
                f.write(f"{i}. [{verification.status.value.upper()}] {verification.claim.text}\n")
                f.write(f"   Confidence: {verification.confidence_score:.2%}\n")
                f.write(f"   Explanation: {verification.explanation}\n\n")
    
    else:
        raise ValueError(f"Unknown export format: {format}")


def get_statistics(report: HallucinationReport) -> Dict:
    """Get statistics from hallucination report.
    
    Args:
        report: HallucinationReport.
        
    Returns:
        Dictionary of statistics.
    """
    summary = report.get_hallucination_summary()
    total = len(report.verifications)
    
    stats = {
        "total_claims": total,
        "supported": summary["supported"],
        "contradicted": summary["contradicted"],
        "unverifiable": summary["unverifiable"],
        "not_checked": summary["not_checked"],
        "hallucination_score": round(report.hallucination_score, 3),
        "risk_level": report.risk_level.value,
        "hallucination_percentage": round(
            100 * (summary["contradicted"] + summary["unverifiable"]) / total
            if total > 0 else 0,
            1
        ),
        "verifiable_percentage": round(
            100 * summary["supported"] / total
            if total > 0 else 0,
            1
        ),
    }
    
    return stats


def compare_reports(report1: HallucinationReport, report2: HallucinationReport) -> Dict:
    """Compare two hallucination reports.
    
    Args:
        report1: First report.
        report2: Second report.
        
    Returns:
        Comparison dictionary.
    """
    stats1 = get_statistics(report1)
    stats2 = get_statistics(report2)
    
    comparison = {
        "report1_score": stats1["hallucination_score"],
        "report2_score": stats2["hallucination_score"],
        "score_difference": round(stats2["hallucination_score"] - stats1["hallucination_score"], 3),
        "report1_risk": stats1["risk_level"],
        "report2_risk": stats2["risk_level"],
        "risk_changed": stats1["risk_level"] != stats2["risk_level"],
    }
    
    return comparison
