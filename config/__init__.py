"""Configuration module."""

from .config import (
    DetectorConfig,
    ClaimExtractionConfig,
    EvidenceRetrievalConfig,
    InferenceConfig,
    DEFAULT_CONFIG,
    DEFAULT_CLAIM_CONFIG,
    DEFAULT_EVIDENCE_CONFIG,
    DEFAULT_INFERENCE_CONFIG,
)

__all__ = [
    "DetectorConfig",
    "ClaimExtractionConfig",
    "EvidenceRetrievalConfig",
    "InferenceConfig",
    "DEFAULT_CONFIG",
    "DEFAULT_CLAIM_CONFIG",
    "DEFAULT_EVIDENCE_CONFIG",
    "DEFAULT_INFERENCE_CONFIG",
]
