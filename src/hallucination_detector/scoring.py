"""Module for hallucination risk scoring."""

from typing import List, Dict
from .data_models import (
    ClaimVerification,
    VerificationStatus,
    RiskLevel,
)
from config.config import DetectorConfig


class HallucinationScorer:
    """Scores hallucination risk for verified claims."""
    
    def __init__(self, config: DetectorConfig = None):
        """Initialize hallucination scorer.
        
        Args:
            config: Configuration for scoring.
        """
        self.config = config or DetectorConfig()
    
    def score_verification(self, verification: ClaimVerification) -> None:
        """Score a single claim verification.
        
        Args:
            verification: Claim verification to score.
        """
        status = verification.status
        
        if status == VerificationStatus.SUPPORTED:
            verification.confidence_score = 0.0  # No hallucination
        elif status == VerificationStatus.CONTRADICTED:
            verification.confidence_score = 1.0  # High hallucination
        elif status == VerificationStatus.UNVERIFIABLE:
            verification.confidence_score = 0.5  # Medium hallucination
        else:
            verification.confidence_score = 0.3  # Unknown - slight concern
    
    def calculate_overall_score(
        self,
        verifications: List[ClaimVerification]
    ) -> float:
        """Calculate overall hallucination score.
        
        Args:
            verifications: List of verified claims.
            
        Returns:
            Overall hallucination score between 0 and 1.
        """
        if not verifications:
            return 0.0
        
        total_score = sum(v.confidence_score for v in verifications)
        return total_score / len(verifications)
    
    def get_risk_level(self, hallucination_score: float) -> RiskLevel:
        """Determine risk level from hallucination score.
        
        Args:
            hallucination_score: Hallucination score between 0 and 1.
            
        Returns:
            Risk level classification.
        """
        if hallucination_score < 0.25:
            return RiskLevel.LOW
        elif hallucination_score < 0.65:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH
    
    def generate_summary(self, verifications: List[ClaimVerification]) -> str:
        """Generate a human-readable summary of verification results.
        
        Args:
            verifications: List of verified claims.
            
        Returns:
            Summary string.
        """
        if not verifications:
            return "No claims were verified."
        
        summary_dict = {
            "supported": 0,
            "contradicted": 0,
            "unverifiable": 0,
        }
        
        for v in verifications:
            if v.status == VerificationStatus.SUPPORTED:
                summary_dict["supported"] += 1
            elif v.status == VerificationStatus.CONTRADICTED:
                summary_dict["contradicted"] += 1
            elif v.status == VerificationStatus.UNVERIFIABLE:
                summary_dict["unverifiable"] += 1
        
        total = len(verifications)
        total_hallucination = summary_dict["contradicted"] + summary_dict["unverifiable"]
        
        summary = (
            f"Out of {total} claims verified: "
            f"{summary_dict['supported']} supported, "
            f"{summary_dict['contradicted']} contradicted, "
            f"{summary_dict['unverifiable']} unverifiable. "
            f"Total potential hallucinations: {total_hallucination} "
            f"({100 * total_hallucination / total:.1f}%)"
        )
        
        return summary
    
    def get_problem_claims(
        self,
        verifications: List[ClaimVerification]
    ) -> List[Dict]:
        """Get problematic claims with details.
        
        Args:
            verifications: List of verified claims.
            
        Returns:
            List of problematic claims with their details.
        """
        problems = []
        
        for v in verifications:
            if v.status in [VerificationStatus.CONTRADICTED, VerificationStatus.UNVERIFIABLE]:
                problem = {
                    "claim": v.claim.text,
                    "status": v.status.value,
                    "confidence": v.confidence_score,
                    "explanation": v.explanation,
                    "evidence": {
                        "supporting": [
                            {
                                "source": e.source,
                                "text": e.text[:100] + "...",
                                "relevance": e.relevance_score
                            }
                            for e in v.supporting_evidence[:2]
                        ],
                        "contradicting": [
                            {
                                "source": e.source,
                                "text": e.text[:100] + "...",
                                "relevance": e.relevance_score
                            }
                            for e in v.contradicting_evidence[:2]
                        ]
                    }
                }
                problems.append(problem)
        
        return problems
