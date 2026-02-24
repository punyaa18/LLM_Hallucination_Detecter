"""Main hallucination detection module."""

from typing import List, Optional
import json

from .claim_extractor import ClaimExtractor
from .evidence_retriever import EvidenceRetriever
from .inference_model import InferenceModel
from .scoring import HallucinationScorer
from .data_models import (
    Claim,
    ClaimVerification,
    Evidence,
    HallucinationReport,
    VerificationStatus,
)
from config.config import DetectorConfig


class HallucinationDetector:
    """Main hallucination detection system."""
    
    def __init__(self, config: DetectorConfig = None):
        """Initialize hallucination detector.
        
        Args:
            config: Configuration for the detector.
        """
        self.config = config or DetectorConfig()
        
        # Initialize components
        self.claim_extractor = ClaimExtractor()
        self.evidence_retriever = EvidenceRetriever(
            embedding_model=self.config.embedding_model_name
        )
        self.inference_model = InferenceModel(
            nli_model_name=self.config.nli_model_name,
            device=self.config.nli_device
        )
        self.scorer = HallucinationScorer(config)
    
    def detect(self, text: str) -> HallucinationReport:
        """Detect hallucinations in the given text.
        
        Args:
            text: Text to analyze for hallucinations.
            
        Returns:
            HallucinationReport with full analysis.
        """
        if self.config.verbose:
            print(f"Analyzing text: {text[:100]}...")
        
        # Step 1: Extract claims
        if self.config.verbose:
            print("Step 1: Extracting claims...")
        claims = self.claim_extractor.extract_claims(
            text,
            max_claims=self.config.max_claims_per_text
        )
        
        if self.config.verbose:
            print(f"  Found {len(claims)} claims")
        
        # Step 2: Retrieve evidence for each claim
        if self.config.verbose:
            print("Step 2: Retrieving evidence...")
        verifications = []
        for i, claim in enumerate(claims):
            if self.config.verbose:
                print(f"  Processing claim {i+1}/{len(claims)}: {claim.text[:50]}...")
            
            verification = self._verify_claim(claim)
            verifications.append(verification)
        
        # Step 3: Score hallucinations
        if self.config.verbose:
            print("Step 3: Scoring hallucinations...")
        
        for verification in verifications:
            self.scorer.score_verification(verification)
        
        # Step 4: Generate report
        if self.config.verbose:
            print("Step 4: Generating report...")
        
        overall_score = self.scorer.calculate_overall_score(verifications)
        risk_level = self.scorer.get_risk_level(overall_score)
        summary = self.scorer.generate_summary(verifications)
        
        report = HallucinationReport(
            original_text=text,
            claims=claims,
            verifications=verifications,
            hallucination_score=overall_score,
            risk_level=risk_level,
            summary=summary,
        )
        
        if self.config.verbose:
            print(f"  Analysis complete. Hallucination score: {overall_score:.2f}")
        
        return report
    
    def _verify_claim(self, claim: Claim) -> ClaimVerification:
        """Verify a single claim against evidence.
        
        Args:
            claim: Claim to verify.
            
        Returns:
            ClaimVerification with verification results.
        """
        verification = ClaimVerification(claim=claim)
        
        # Retrieve evidence
        evidence_list = self.evidence_retriever.retrieve_evidence(
            claim,
            num_results=self.config.max_evidence_per_claim
        )
        
        if not evidence_list:
            verification.status = VerificationStatus.UNVERIFIABLE
            verification.explanation = "No supporting evidence found"
            return verification
        
        # Check NLI for each evidence
        entailment_scores = []
        contradiction_scores = []
        neutral_scores = []
        
        for evidence in evidence_list:
            label, confidence = self.inference_model.check_nli(
                premise=evidence.text,
                hypothesis=claim.text
            )
            
            if label == "entailment":
                entailment_scores.append(confidence)
                evidence.relevance_score = confidence
                verification.supporting_evidence.append(evidence)
            elif label == "contradiction":
                contradiction_scores.append(confidence)
                evidence.relevance_score = confidence
                verification.contradicting_evidence.append(evidence)
            else:  # neutral
                neutral_scores.append(confidence)
        
        # Determine verification status
        max_entailment = max(entailment_scores) if entailment_scores else 0
        max_contradiction = max(contradiction_scores) if contradiction_scores else 0
        max_neutral = max(neutral_scores) if neutral_scores else 0
        
        if max_contradiction > self.config.contradicted_threshold:
            verification.status = VerificationStatus.CONTRADICTED
            verification.confidence_score = max_contradiction
            verification.explanation = (
                f"Evidence contradicts the claim with confidence {max_contradiction:.2f}"
            )
        elif max_entailment > self.config.supported_threshold:
            verification.status = VerificationStatus.SUPPORTED
            verification.confidence_score = max_entailment
            verification.explanation = (
                f"Evidence supports the claim with confidence {max_entailment:.2f}"
            )
        else:
            verification.status = VerificationStatus.UNVERIFIABLE
            verification.confidence_score = 0.5
            verification.explanation = (
                "Evidence is unclear or neutral regarding the claim"
            )
        
        return verification
    
    def detect_with_output(
        self,
        text: str,
        output_format: str = "json"
    ) -> str:
        """Detect hallucinations and return formatted output.
        
        Args:
            text: Text to analyze.
            output_format: Format for output ('json' or 'text').
            
        Returns:
            Formatted hallucination report.
        """
        report = self.detect(text)
        
        if output_format == "json":
            return self._format_json(report)
        elif output_format == "text":
            return self._format_text(report)
        else:
            raise ValueError(f"Unknown output format: {output_format}")
    
    def _format_json(self, report: HallucinationReport) -> str:
        """Format report as JSON.
        
        Args:
            report: HallucinationReport to format.
            
        Returns:
            JSON string representation.
        """
        data = {
            "hallucination_score": round(report.hallucination_score, 3),
            "risk_level": report.risk_level.value,
            "summary": report.summary,
            "total_claims": len(report.claims),
            "claims_summary": report.get_hallucination_summary(),
            "problematic_claims": self.scorer.get_problem_claims(report.verifications),
        }
        return json.dumps(data, indent=2)
    
    def _format_text(self, report: HallucinationReport) -> str:
        """Format report as human-readable text.
        
        Args:
            report: HallucinationReport to format.
            
        Returns:
            Text representation.
        """
        lines = [
            "=" * 80,
            "HALLUCINATION DETECTION REPORT",
            "=" * 80,
            "",
            f"Overall Hallucination Score: {report.hallucination_score:.2%}",
            f"Risk Level: {report.risk_level.value.upper()}",
            "",
            f"Summary: {report.summary}",
            "",
            "PROBLEMATIC CLAIMS:",
            "-" * 80,
        ]
        
        problematic = report.get_problematic_claims()
        if problematic:
            for i, verification in enumerate(problematic, 1):
                lines.append(f"\n{i}. [{verification.status.value.upper()}] {verification.claim.text}")
                lines.append(f"   Confidence: {verification.confidence_score:.2%}")
                lines.append(f"   Explanation: {verification.explanation}")
                
                if verification.contradicting_evidence:
                    lines.append(f"   Contradicting Evidence:")
                    for evidence in verification.contradicting_evidence[:1]:
                        lines.append(f"     - {evidence.source}: {evidence.text[:150]}...")
        else:
            lines.append("None - all verified claims are supported.")
        
        lines.extend(["", "=" * 80])
        
        return "\n".join(lines)
