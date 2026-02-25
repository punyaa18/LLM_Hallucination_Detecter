"""Ollama HTTP client helpers."""

from typing import Any, Dict, List, Optional
import requests


class OllamaClient:
    """Minimal Ollama HTTP client."""

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        num_predict: int = 128,
        system: Optional[str] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        if system:
            payload["system"] = system

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")

    def embeddings(self, model: str, text: str) -> List[float]:
        payload = {
            "model": model,
            "prompt": text,
        }
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("embedding", [])


class OllamaEmbeddingModel:
    """Embedding wrapper that matches SentenceTransformer encode()."""

    def __init__(self, client: OllamaClient, model: str):
        self.client = client
        self.model = model

    def encode(self, text: str) -> List[float]:
        return self.client.embeddings(self.model, text)