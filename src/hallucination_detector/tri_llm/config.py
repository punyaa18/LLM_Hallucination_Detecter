"""Configuration and data contracts for tri-LLM benchmarking."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class TriLLMThresholds:
    """Tunable thresholds for scoring and categorical risk."""

    high_risk_score: float = 70.0
    medium_risk_score: float = 40.0
    severe_divergence_threshold: float = 0.45
    overconfidence_marker_threshold: float = 55.0


@dataclass
class TriLLMConfig:
    """Runtime config for tri-LLM benchmark."""

    openai_model: str
    ollama_model_a: str
    ollama_model_b: str
    prompts: List[str]

    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_base_url: str = "http://localhost:11434"

    request_timeout_seconds: int = 90
    generation_temperature: float = 0.2
    generation_max_tokens: int = 512

    embedding_model_name: str = "all-MiniLM-L6-v2"

    thresholds: TriLLMThresholds = field(default_factory=TriLLMThresholds)
    output_path: str = "tri_llm_benchmark_results.json"
    log_level: str = "INFO"


@dataclass
class ModelResponse:
    """Single model answer for a prompt."""

    model_key: str
    provider: str
    model_name: str
    prompt: str
    response: str
    latency_seconds: float
    error: Optional[str] = None


@dataclass
class PairSimilarity:
    """Semantic similarity between two models for one prompt."""

    model_x: str
    model_y: str
    score: float


@dataclass
class PromptModelEvaluation:
    """Per-model metrics for one prompt."""

    model_key: str
    consistency_score: float
    hallucination_risk: str
    overconfidence_score: float
    estimated_incorrectness_level: str
    divergence_flags: List[str]
    evidence: Dict[str, object]


@dataclass
class PromptEvaluation:
    """All per-prompt information across models."""

    prompt_index: int
    prompt: str
    responses: Dict[str, ModelResponse]
    pairwise_similarity: List[PairSimilarity]
    per_model: Dict[str, PromptModelEvaluation]


def utc_timestamp() -> str:
    """Stable UTC timestamp for report metadata."""
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
