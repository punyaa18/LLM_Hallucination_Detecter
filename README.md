# LLM Hallucination Detector
 Wrote a lil something :p -- https://medium.com/@punyaa184/why-ai-hallucinations-are-dangerous-349a5954cd34                  

 
Detect unsupported, contradictory, and unverifiable claims in LLM-generated text. The project combines claim extraction, evidence retrieval, NLI verification, and scoring in a small local toolkit with both Streamlit and Flask entrypoints.

![Pipeline diagram](Hallucinations%20in%20Local%20Large%20Language%20Models/figures/feature_architecture.png)

![Consistency chart](Hallucinations%20in%20Local%20Large%20Language%20Models/figures/consistency_scores.png)

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the UI with `streamlit run web_app.py` or start the API with `PYTHONPATH=src python -m hallucination_detector.web_app`.

## Use It

```python
from config.config import DetectorConfig
from hallucination_detector import HallucinationDetector

report = HallucinationDetector(DetectorConfig(verbose=False)).detect("The Moon is made of cheese.")
print(report.summary)
```

## Configuration

The main knobs are in `config/config.py`: `nli_backend`, `embedding_backend`, `ollama_base_url`, `supported_threshold`, `contradicted_threshold`, `max_claims_per_text`, and `max_evidence_per_claim`.

## Examples and Tests

See `examples/` for basic, advanced, interactive, comparison, and tri-LLM benchmark scripts. Run `pytest` to execute the test suite.
