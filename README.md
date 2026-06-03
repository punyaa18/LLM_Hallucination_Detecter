# LLM Hallucination Detector
 Wrote a lil something :p -- https://medium.com/@punyaa184/why-ai-hallucinations-are-dangerous-349a5954cd34                  

Detect unsupported, contradictory, and unverifiable claims in LLM-generated text. The project combines claim extraction, evidence retrieval, NLI verification, hallucination scoring, and a modern research-style dashboard for visual inspection of results.

<a href="Hallucinations%20in%20Local%20Large%20Language%20Models/figures/consistency_scores.png" target="_blank" rel="noreferrer">
	<img width="1024" height="525" alt="image" src="https://github.com/user-attachments/assets/38e9c508-3b16-4cae-b80e-4e16859c8845" />

</a>

## What It Does

The detection stack is built around four stages:

1. Claim extraction from model output.
2. Evidence retrieval from trusted sources such as Wikipedia.
3. NLI verification using either Hugging Face models or Ollama-backed Llama 3 / Mistral configurations.
4. Hallucination scoring and risk labeling.

The Flask dashboard now renders a dark blue/purple results screen with a claim panel, evidence retrieval panel, verification verdicts, and a high-contrast hallucination score card.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the dashboard with `PYTHONPATH=src python -m hallucination_detector.web_app`. The UI is available at `http://127.0.0.1:8080`.

## Example

Sample input:

```text
Paris is the capital of Germany
```

The dashboard highlights the extracted claim, retrieved evidence that Germany's capital is Berlin, the contradiction verdict, and the resulting hallucination risk score.

## Use It

```python
from config.config import DetectorConfig
from hallucination_detector import HallucinationDetector

report = HallucinationDetector(DetectorConfig(verbose=False)).detect("The Moon is made of cheese.")
print(report.summary)
```

## Configuration

The main knobs are in `config/config.py`: `nli_backend`, `embedding_backend`, `ollama_base_url`, `ollama_nli_model`, `ollama_embedding_model`, `supported_threshold`, `contradicted_threshold`, `max_claims_per_text`, and `max_evidence_per_claim`.

## Examples and Tests

See `examples/` for basic, advanced, interactive, comparison, and tri-LLM benchmark scripts. Run `pytest` to execute the test suite.
