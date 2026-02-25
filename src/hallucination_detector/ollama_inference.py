"""Ollama-backed NLI inference."""

from typing import Tuple
import json
import re

from .ollama_client import OllamaClient
from config.config import InferenceConfig


class OllamaInferenceModel:
    """Performs NLI using a local Ollama model."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        timeout: int = 30,
        temperature: float = 0.0,
        max_tokens: int = 128,
        config: InferenceConfig = None,
    ):
        self.model_name = model_name
        self.client = OllamaClient(base_url=base_url, timeout=timeout)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.config = config or InferenceConfig()

    def check_nli(self, premise: str, hypothesis: str) -> Tuple[str, float]:
        """Check NLI using Ollama.

        Returns:
            Tuple of (label, confidence) where label is entailment/contradiction/neutral.
        """
        prompt = (
            "You are a strict fact-checking model. "
            "Compare the evidence (premise) to the claim (hypothesis). "
            "Return ONLY valid JSON with keys: label and confidence. "
            "label must be one of: entailment, contradiction, neutral. "
            "confidence must be a number between 0 and 1.\n\n"
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n"
            "JSON:"
        )

        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            num_predict=self.max_tokens,
            system="Return only JSON, no extra text.",
        )

        label, confidence = self._parse_response(response)
        return label, confidence

    def _parse_response(self, response: str) -> Tuple[str, float]:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            return "neutral", 0.33

        try:
            data = json.loads(match.group(0))
            raw_label = str(data.get("label", "neutral")).strip().lower()
            confidence = float(data.get("confidence", 0.33))
        except Exception:
            return "neutral", 0.33

        label_map = {
            "entailment": "entailment",
            "supported": "entailment",
            "support": "entailment",
            "contradiction": "contradiction",
            "contradicted": "contradiction",
            "neutral": "neutral",
            "unverifiable": "neutral",
            "unknown": "neutral",
        }
        label = label_map.get(raw_label, "neutral")

        if confidence < 0:
            confidence = 0.0
        if confidence > 1:
            confidence = 1.0

        return label, confidence