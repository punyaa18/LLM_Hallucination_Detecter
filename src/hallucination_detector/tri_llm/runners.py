"""Model runners for OpenAI and Ollama backends."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict

import requests

from .config import ModelResponse, TriLLMConfig

LOGGER = logging.getLogger(__name__)


class RunnerError(RuntimeError):
    """Raised when a model backend call fails."""


@dataclass
class OpenAIRunner:
    """Runner for OpenAI-compatible chat completions endpoint."""

    config: TriLLMConfig

    def generate(self, prompt: str) -> ModelResponse:
        started = time.perf_counter()
        headers = {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.generation_temperature,
            "max_tokens": self.config.generation_max_tokens,
        }

        response = requests.post(
            f"{self.config.openai_base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.config.request_timeout_seconds,
        )
        latency = time.perf_counter() - started

        if response.status_code >= 400:
            raise RunnerError(f"OpenAI request failed: {response.status_code} {response.text[:300]}")

        body = response.json()
        text = body["choices"][0]["message"]["content"].strip()
        return ModelResponse(
            model_key="openai",
            provider="openai",
            model_name=self.config.openai_model,
            prompt=prompt,
            response=text,
            latency_seconds=round(latency, 3),
        )


@dataclass
class OllamaRunner:
    """Runner for local Ollama generate endpoint."""

    config: TriLLMConfig
    model_key: str
    model_name: str

    def generate(self, prompt: str) -> ModelResponse:
        started = time.perf_counter()
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.generation_temperature,
                "num_predict": self.config.generation_max_tokens,
            },
        }
        response = requests.post(
            f"{self.config.ollama_base_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=self.config.request_timeout_seconds,
        )
        latency = time.perf_counter() - started

        if response.status_code >= 400:
            raise RunnerError(f"Ollama request failed ({self.model_name}): {response.status_code} {response.text[:300]}")

        body = response.json()
        text = body.get("response", "").strip()
        return ModelResponse(
            model_key=self.model_key,
            provider="ollama",
            model_name=self.model_name,
            prompt=prompt,
            response=text,
            latency_seconds=round(latency, 3),
        )


class TriModelRunner:
    """Executes three model calls for each prompt."""

    def __init__(self, config: TriLLMConfig):
        self.config = config
        self.openai = OpenAIRunner(config)
        self.ollama_a = OllamaRunner(config, model_key="ollama_a", model_name=config.ollama_model_a)
        self.ollama_b = OllamaRunner(config, model_key="ollama_b", model_name=config.ollama_model_b)

    def run_prompt(self, prompt: str) -> Dict[str, ModelResponse]:
        """Run all configured models on a single prompt."""
        results: Dict[str, ModelResponse] = {}
        for runner in (self.openai, self.ollama_a, self.ollama_b):
            try:
                result = runner.generate(prompt)
                LOGGER.info("Model %s completed in %.3fs", result.model_key, result.latency_seconds)
                results[result.model_key] = result
            except Exception as error:
                model_key = "openai" if runner is self.openai else runner.model_key
                model_name = self.config.openai_model if runner is self.openai else runner.model_name
                LOGGER.exception("Model %s failed", model_key)
                results[model_key] = ModelResponse(
                    model_key=model_key,
                    provider="openai" if model_key == "openai" else "ollama",
                    model_name=model_name,
                    prompt=prompt,
                    response="",
                    latency_seconds=0.0,
                    error=str(error),
                )
        return results
