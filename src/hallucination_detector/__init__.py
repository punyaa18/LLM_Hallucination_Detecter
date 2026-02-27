"""LLM Hallucination Detector package."""

from importlib import import_module

__version__ = "1.0.0"

_EXPORTS = {
    "HallucinationDetector": (".detector", "HallucinationDetector"),
    "ClaimExtractor": (".claim_extractor", "ClaimExtractor"),
    "EvidenceRetriever": (".evidence_retriever", "EvidenceRetriever"),
    "InferenceModel": (".inference_model", "InferenceModel"),
    "OllamaInferenceModel": (".ollama_inference", "OllamaInferenceModel"),
    "HallucinationScorer": (".scoring", "HallucinationScorer"),
    "Claim": (".data_models", "Claim"),
    "Evidence": (".data_models", "Evidence"),
    "ClaimVerification": (".data_models", "ClaimVerification"),
    "HallucinationReport": (".data_models", "HallucinationReport"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
