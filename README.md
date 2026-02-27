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

## Tri-LLM hallucination benchmark (OpenAI + 2 local Ollama models)

This framework sends each prompt to three models:
- OpenAI model (API)
- Ollama Model A (local)
- Ollama Model B (local)

For each model per prompt it computes:
- Consistency score (0-100)
- Hallucination risk (Low/Medium/High)
- Overconfidence score (0-100)
- Estimated incorrectness level

It also detects:
- Factual divergence and cross-model contradiction patterns
- Fabricated details and unsupported specific claims
- Overconfident tone without uncertainty markers
- Cases where a model should have said "I don't know" but speculated

Run benchmark:

```bash
python examples/tri_llm_benchmark.py \
	--openai-model gpt-4o-mini \
	--ollama-model-a llama3.2:latest \
	--ollama-model-b mistral:latest \
	--prompts-file examples/prompts.txt \
	--num-prompts 5 \
	--high-risk-threshold 70 \
	--medium-risk-threshold 40 \
	--output tri_llm_benchmark_results.json
```

Alternative prompt input:

```bash
python examples/tri_llm_benchmark.py \
	--openai-model gpt-4o-mini \
	--ollama-model-a llama3.2:latest \
	--ollama-model-b mistral:latest \
	--prompts-json '["Prompt 1", "Prompt 2"]'
```

Output includes:
- Full prompt-by-prompt responses and metrics in JSON
- Final benchmark summary and most reliable model
- Hallucination and overconfidence frequencies
- Analytical conclusion and mitigation strategies (RAG grounding, calibrated refusal, confidence gating)

Run tri-LLM unit tests:

```bash
pytest tests/test_tri_llm_framework.py -q
```
