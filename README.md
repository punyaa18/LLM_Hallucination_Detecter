# LLM_Hallucination_Detecter
part of 8th sem project 

## Compare two local Ollama models (plus optional online model)

Run this script from project root:

```bash
python examples/compare_model_hallucinations.py --model-a llama3.2:latest --model-b mistral:latest
```

With online baseline:       

```bash
python examples/compare_model_hallucinations.py --model-a llama3.2:latest --model-b mistral:latest --online-provider openrouter --online-model openai/gpt-4o-mini
```

Key outputs:
- Average hallucination score per model
- Hallucination rate per prompt
- First prompt index where model starts hallucinating (based on threshold)
- Speed (latency) and claim verification breakdown
- Preference notes for local vs online trade-offs

The script writes full details to `benchmark_results.json` by default.

## Visual dashboard (HTML + CSS)

Run a local dashboard to watch if the model is moving toward hallucination:

```bash
python -m hallucination_detector.web_app
```

Then open: `http://127.0.0.1:8080`

What it shows:
- Hallucination score
- Risk level
- Early warning banner (`safe`, `watch`, `danger`) for "about to hallucinate"
- Claim-wise status table (supported / contradicted / unverifiable)

## Prepare visual data

Create a seed dataset for UI testing:

```bash
python examples/prepare_visual_data.py
```

Output: `data/visual_seed_dataset.jsonl`

Optional: run live detection and save detected warning labels:

```bash
python examples/prepare_visual_data.py --analyze-live --nli-backend ollama
```

## Run tests

```bash
pytest tests/test_visual_signals.py tests/test_prepare_visual_data.py -q
```
