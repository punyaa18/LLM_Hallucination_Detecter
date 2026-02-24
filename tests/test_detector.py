"""Tests for hallucination detector."""

import sys
sys.path.insert(0, "/workspaces/LLM_Hallucination_Detecter/src")

import pytest
from hallucination_detector import HallucinationDetector
from hallucination_detector.data_models import VerificationStatus, RiskLevel
from config import DetectorConfig


class TestHallucinationDetector:
    """Test suite for HallucinationDetector."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance for testing."""
        config = DetectorConfig(verbose=False, nli_device="cpu")
        return HallucinationDetector(config)
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        config = DetectorConfig()
        detector = HallucinationDetector(config)
        assert detector is not None
        assert detector.claim_extractor is not None
        assert detector.evidence_retriever is not None
        assert detector.inference_model is not None
    
    def test_empty_text(self, detector):
        """Test with empty text."""
        report = detector.detect("")
        assert report is not None
        assert len(report.claims) == 0
        assert len(report.verifications) == 0
    
    def test_simple_claim_extraction(self, detector):
        """Test claim extraction from simple text."""
        text = "The Earth is approximately 4.5 billion years old."
        report = detector.detect(text)
        
        assert len(report.claims) > 0
        assert report.claims[0].text is not None
    
    def test_hallucination_score_range(self, detector):
        """Test hallucination score is between 0 and 1."""
        text = "Python was created by Guido van Rossum in 1991. It is green and can fly."
        report = detector.detect(text)
        
        assert 0 <= report.hallucination_score <= 1
    
    def test_risk_level_assignment(self, detector):
        """Test risk level is correctly assigned."""
        text = "The sky is green and made of ice."
        report = detector.detect(text)
        
        assert report.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]
    
    def test_verification_status(self, detector):
        """Test verification status is assigned."""
        text = "London is the capital of England."
        report = detector.detect(text)
        
        for verification in report.verifications:
            assert verification.status in [
                VerificationStatus.SUPPORTED,
                VerificationStatus.CONTRADICTED,
                VerificationStatus.UNVERIFIABLE,
                VerificationStatus.NOT_CHECKED
            ]
    
    def test_report_summary(self, detector):
        """Test report summary generation."""
        text = "Rome is the capital of Italy. Italy is in Europe."
        report = detector.detect(text)
        
        assert report.summary is not None
        assert len(report.summary) > 0
    
    def test_json_output_format(self, detector):
        """Test JSON format output."""
        text = "The Eiffel Tower is in Paris."
        output = detector.detect_with_output(text, output_format="json")
        
        assert output is not None
        assert '"hallucination_score"' in output or "'hallucination_score'" in output
    
    def test_text_output_format(self, detector):
        """Test text format output."""
        text = "The Statue of Liberty is in New York."
        output = detector.detect_with_output(text, output_format="text")
        
        assert output is not None
        assert "HALLUCINATION" in output
    
    def test_multiple_claims(self, detector):
        """Test detection with multiple claims."""
        text = """
        The Great Wall of China stretches for over 13,000 miles.
        It was built to protect from invasions.
        The wall is visible from space.
        """
        report = detector.detect(text)
        
        assert len(report.claims) > 0
    
    def test_problematic_claims_extraction(self, detector):
        """Test extraction of problematic claims."""
        text = "Paris is the capital of France. Water freezes at 0 Celsius. Cats are reptiles."
        report = detector.detect(text)
        
        problematic = report.get_problematic_claims()
        # Should have some unverifiable or contradicted claims
        assert isinstance(problematic, list)


class TestClaimExtractor:
    """Test suite for ClaimExtractor."""
    
    def test_claim_extraction_basic(self):
        """Test basic claim extraction."""
        from hallucination_detector import ClaimExtractor
        
        extractor = ClaimExtractor()
        text = "The Moon orbits the Earth."
        claims = extractor.extract_claims(text)
        
        assert len(claims) > 0
    
    def test_claim_deduplication(self):
        """Test claim deduplication."""
        from hallucination_detector import ClaimExtractor
        
        extractor = ClaimExtractor()
        text = "The Moon orbits the Earth. The Moon orbits the Earth."
        claims = extractor.extract_claims(text)
        
        # Should not have exact duplicates
        claim_texts = [c.text for c in claims]
        assert len(claim_texts) == len(set(claim_texts))


class TestEvidence:
    """Test suite for Evidence retrieval."""
    
    def test_wikipedia_retrieval(self):
        """Test Wikipedia evidence retrieval."""
        from hallucination_detector import EvidenceRetriever
        from hallucination_detector.data_models import Claim
        
        retriever = EvidenceRetriever()
        claim = Claim(
            text="Paris is the capital of France",
            start_idx=0,
            end_idx=30
        )
        
        evidence = retriever.retrieve_evidence(claim)
        # Should return some evidence
        assert isinstance(evidence, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
