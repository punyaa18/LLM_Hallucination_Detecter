"""LLM Hallucination Detector package."""

from .detector import HallucinationDetector
from .claim_extractor import ClaimExtractor
from .evidence_retriever import EvidenceRetriever
from .inference_model import InferenceModel
from .ollama_inference import OllamaInferenceModel
from .scoring import HallucinationScorer
from .data_models import (
    Claim,
    Evidence,
    ClaimVerification,
    HallucinationReport,
)

__version__ = "1.0.0"
__all__ = [
    "HallucinationDetector",
    "ClaimExtractor",
    "EvidenceRetriever",
    "InferenceModel",
    "OllamaInferenceModel",
    "HallucinationScorer",
    "Claim",
    "Evidence",
    "ClaimVerification",
    "HallucinationReport",
]
