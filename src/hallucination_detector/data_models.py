"""Data models for hallucination detection."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class VerificationStatus(str, Enum):
    """Status of claim verification."""
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"
    NOT_CHECKED = "not_checked"


class RiskLevel(str, Enum):
    """Risk level of hallucination."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Evidence:
    """Evidence for/against a claim."""
    
    source: str
    text: str
    url: Optional[str] = None
    relevance_score: float = 0.0
    
    def __repr__(self) -> str:
        return f"Evidence(source={self.source}, relevance={self.relevance_score:.2f})"


@dataclass
class Claim:
    """Extracted factual claim from text."""
    
    text: str
    start_idx: int
    end_idx: int
    confidence: float = 0.0
    
    def __repr__(self) -> str:
        return f"Claim(text='{self.text[:50]}...', confidence={self.confidence:.2f})"


@dataclass
class ClaimVerification:
    """Verification result for a claim."""
    
    claim: Claim
    status: VerificationStatus = VerificationStatus.NOT_CHECKED
    confidence_score: float = 0.0
    supporting_evidence: List[Evidence] = field(default_factory=list)
    contradicting_evidence: List[Evidence] = field(default_factory=list)
    reasoning: str = ""
    explanation: str = ""
    
    def get_best_evidence(self) -> Optional[Evidence]:
        """Get the best quality evidence."""
        all_evidence = self.supporting_evidence + self.contradicting_evidence
        if not all_evidence:
            return None
        return max(all_evidence, key=lambda e: e.relevance_score)
    
    def __repr__(self) -> str:
        return (
            f"ClaimVerification("
            f"claim='{self.claim.text[:30]}...', "
            f"status={self.status.value}, "
            f"confidence={self.confidence_score:.2f})"
        )


@dataclass
class HallucinationReport:
    """Complete hallucination detection report."""
    
    original_text: str
    claims: List[Claim] = field(default_factory=list)
    verifications: List[ClaimVerification] = field(default_factory=list)
    hallucination_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_hallucination_summary(self) -> Dict[str, int]:
        """Get count of claims by verification status."""
        summary = {
            "supported": 0,
            "contradicted": 0,
            "unverifiable": 0,
            "not_checked": 0,
        }
        for v in self.verifications:
            summary[v.status.value] += 1
        return summary
    
    def get_problematic_claims(self) -> List[ClaimVerification]:
        """Get claims that are contradicted or unverifiable."""
        return [
            v for v in self.verifications
            if v.status in [VerificationStatus.CONTRADICTED, VerificationStatus.UNVERIFIABLE]
        ]
    
    def __repr__(self) -> str:
        return (
            f"HallucinationReport("
            f"claims={len(self.claims)}, "
            f"hallucination_score={self.hallucination_score:.2f}, "
            f"risk_level={self.risk_level.value})"
        )
