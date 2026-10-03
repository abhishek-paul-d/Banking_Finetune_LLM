# Project Plan: Self-Hosted Banking Triage LLM

Goal: fine-tune a small open model on Banking77 to return strict JSON triage
(`intent`, `urgency`, `needs_human`), quantize it, serve it, and publish
quality / latency / cost numbers. Work is phased; each phase ends with a
decision or a deliverable.

Compute: Kaggle (2x T4 16 GB, fp16, 4-bit QLoRA); one model per GPU in parallel. Later phases may need an A100/L4-class GPU for fair serving benchmarks. No AWS.

---

## Phase 0: Prep

**Tasks**
- Verify the data pipeline and label rules (`src/schema.py`) on a Colab run.
- Add `--sample-per-class N` to `src/data/prepare.py` for a stratified pilot subset (all 77 intents represented).
- Make training take a per-model config (`configs/<model>.yaml`); confirm each model's chat template and tokenizer work with the prompt format.

**Done when:** `data/pilot_train.jsonl` exists and a 1-step training smoke test passes for both candidate models.

---

## Phase 1: Pilot comparison of two models (decision phase)

**Candidates (decided)**

| Model | HF id | Notes |
|---|---|---|
| Qwen3.5-4B | `Qwen/Qwen3.5-4B` | Apache-2.0, ungated; multimodal checkpoint, hybrid (Gated DeltaNet) text decoder; thinking ON by default in its chat template |
| Gemma-4-E4B | `google/gemma-4-E4B-it` | Apache-2.0, ungated; multimodal checkpoint (vision + audio towers); "E4B" = ~4B effective params |

Both are loaded text-only. Prompts are rendered once with thinking disabled (`src/formatting.py`)
and reused for training and eval, so the two models see equivalent inputs.

**Design**
- Pilot train set: 2,000 examples (about 26 per intent), identical for both models.
- Everything else identical: hyperparameters, seed, epochs. Only the model changes.
- Evaluate on the full 3,080-example test set.
- Also score each model with no fine-tuning (zero-shot baseline) to show what fine-tuning adds.

**Metrics recorded per model**
- Intent accuracy, macro-F1, JSON-valid rate, urgency accuracy, needs_human accuracy
- Training time, peak VRAM, inference tokens/sec
- Everything logged to MLflow

**Decision rule (fixed before running)**
1. Pick the model with the higher macro-F1.
2. If within about 1 point, pick the faster/smaller one.
3. Winner must reach at least 98% valid JSON.
4. If the result is close, rerun with a second seed before deciding.

**Done when:** a comparison table is in the README and the winner is named.

---

## Phase 2: Full fine-tune of the winner

- Train on all Banking77 train data (about 9k examples).
- Evaluate on the full test set; log to MLflow.
- Save the LoRA adapter and the merged fp16 model.

**Done when:** merged model + metrics saved to Drive and MLflow.

---

## Phase 3: Quantization

- Produce AWQ int4 (`src/quantize/awq_quantize.py`) and GGUF Q4_K_M (`scripts/make_gguf.sh`).
- Compare against fp16: quality (eval metrics), disk size, memory, speed.
- Risk: `autoawq` pins old torch versions; fallback is llm-compressor.

**Done when:** all three variants are scored and sized in one table.

---

## Phase 4: Serving and benchmarks

- Serve fp16 and AWQ with vLLM, GGUF with llama.cpp server.
- Concurrency sweep (1, 4, 8, 16, 32, 64): p50/p95 latency, tokens/sec.
- Cost per 1M tokens from measured throughput and GPU $/hour (state the A100 assumption).
- Compare against a hosted API's per-token price on the same eval set.
- FastAPI gateway (`src/serve/app.py`) validates output; `docker-compose.yml` runs vLLM + gateway on any GPU machine.

**Done when:** `results/bench.csv` and the cost table are complete.

---

## Phase 5: CI eval gate

Built (local parts verified; the GitHub run is the last step):
- `data/gate_slice.jsonl`: fixed, class-balanced 200-example slice (`src/eval/make_gate_slice.py`), committed to the repo.
- `configs/gate.yaml`: floors set from the Q4_K_M predictions on that slice (intent acc 0.96 measured, floor 0.90). The gate catches broken models, not 0.5-point drift (noise on 200 examples is about 1.4 points).
- `src/eval/gate.py`: min and max thresholds, missing metric = fail, job-summary table.
- `tests/`: good predictions pass, un-tuned / wrong-label / invented-label predictions fail; gateway prompt equals the training prompt.
- `.github/workflows/eval-gate.yml`: unit tests -> CPU llama.cpp (cached) -> eval on the slice -> gate -> publish gateway image to GHCR only if everything passed.
- Gateway (`src/serve/app.py`) now sends the same thinking-off prompt as training via `/v1/completions`.

**DONE (green run: fine-tuned Q4_K_M passes; red run: un-tuned base fails 5 of 6 checks).** To repeat, run the workflow manually (Actions > Run workflow) with
repo `unsloth/Qwen3.5-4B-GGUF`, file `Qwen3.5-4B-Q4_K_M.gguf` (the un-tuned base) and keep the red run; then a normal run (green).

---

## Phase 6: Packaging

Done:
- `README.md`: results table (quality, size, latency, throughput, cost), findings, pipeline diagram, run instructions, limitations.
- `docs/FINDINGS.md` (full write-up), `docs/MODEL_CARD.md` (copy to the Hugging Face repo as `README.md`).
- `results/`: every table and CSV behind the numbers; `scripts/cost.py` regenerates the cost table (GPU price $0.156/h = 1.56 Colab units/h at $9.99 per 100 units).
- Cleanup: stale notebooks and `loadtest/bench.py` removed; `docker-compose.yml` updated.

Still open (needs the user): upload `docs/MODEL_CARD.md` to the HF repo; optional hosted-API price row; end-to-end test of gateway + docker-compose.

---

## Phase 1b: second seed for the pilot comparison (decision rule fixed before running)

Repeat the pilot for both models with training seed 1 (`configs/pilot/*_seed1.yaml`, notebook `phase1b_pilot_seed1_kaggle.ipynb`); data and every other setting unchanged.
`src/eval/seed_summary.py` applies: **tie confirmed** if in both seeds |macro-F1 diff| < 1 point or its paired-bootstrap CI includes 0; **a model ahead** if the CI excludes 0 in the same direction in both seeds
(then the model choice is reopened); otherwise **mixed**, reported as a tie. Qwen stays the production model unless Gemma is clearly ahead in both seeds.

**Result (done):** tie confirmed. Qwen minus Gemma macro-F1 = +0.0025 (seed 0) and -0.0042 (seed 1), both CIs include 0; Qwen stays on memory and inference speed. See `results/pilot/seed_summary.md`.
