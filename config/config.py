"""Configuration for hallucination detector."""

from dataclasses import dataclass
from typing import List


@dataclass
class DetectorConfig:
    """Configuration for the hallucination detector."""
    
    # NLI Model configuration
    nli_model_name: str = "roberta-large-mnli"
    nli_device: str = "cpu"
    nli_batch_size: int = 32
    
    # Embedding model configuration
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    
    # Claim extraction
    claim_extraction_model: str = "gpt2"  # Can be replaced with more advanced models
    max_claims_per_text: int = 10
    min_claim_length: int = 10
    
    # Evidence retrieval
    evidence_sources: List[str] = None
    max_evidence_per_claim: int = 3
    evidence_retrieval_timeout: int = 10
    wikipedia_search_limit: int = 3
    
    # Scoring thresholds
    supported_threshold: float = 0.5  # Entailment threshold
    contradicted_threshold: float = 0.5  # Contradiction threshold
    unverifiable_threshold: float = 0.3  # Neutral/unverifiable threshold
    
    # Output configuration
    highlight_claims: bool = True
    include_evidence: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.evidence_sources is None:
            self.evidence_sources = ["wikipedia"]


@dataclass
class ClaimExtractionConfig:
    """Configuration for claim extraction."""
    
    min_tokens: int = 5
    max_tokens: int = 25
    filter_stop_words: bool = True
    extract_named_entities: bool = True


@dataclass
class EvidenceRetrievalConfig:
    """Configuration for evidence retrieval."""
    
    wikipedia_lang: str = "en"
    min_evidence_length: int = 50
    max_evidence_length: int = 500
    fallback_to_summary: bool = True


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    
    # NLI label mapping
    label_mapping: dict = None
    
    # Similarity thresholds
    similarity_threshold: float = 0.7
    
    def __post_init__(self):
        """Initialize default values."""
        if self.label_mapping is None:
            self.label_mapping = {
                0: "entailment",
                1: "neutral",
                2: "contradiction"
            }


# Default configurations
DEFAULT_CONFIG = DetectorConfig()
DEFAULT_CLAIM_CONFIG = ClaimExtractionConfig()
DEFAULT_EVIDENCE_CONFIG = EvidenceRetrievalConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()
