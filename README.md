# Self-Hosted Banking Triage LLM

QLoRA fine-tune of a small open model (Qwen2.5-1.5B-Instruct) on Banking77 to
return strict JSON triage (`intent`, `urgency`, `needs_human`), quantized two
ways (AWQ, GGUF), served with vLLM behind FastAPI, gated by CI evals, and
benchmarked for latency, throughput, and cost per 1M tokens.

> `urgency` and `needs_human` are rule-based labels derived from the intent
> (see `src/schema.py`), not human annotations. Intent labels are Banking77 ground truth.

## Pipeline

```
data/prepare  ->  train/qlora  ->  quantize/merge  ->  awq_quantize | make_gguf.sh
                                        |
          eval/evaluate (per variant) + loadtest/bench (per variant)
                                        |
      serve (vLLM + FastAPI)     CI: eval gate -> deploy
```

## Run it on Colab (A100)

1. Copy this folder to `MyDrive/Banking_FInetuning` in Google Drive.
2. Open `notebooks/01_train_quantize.ipynb` in Colab, set the runtime to A100, run all.
   Produces the fine-tuned model plus AWQ and GGUF variants in `outputs/`.
3. Open `notebooks/02_serve_eval_bench.ipynb` in a fresh A100 runtime, run all.
   Serves each variant, evaluates it, load-tests it, and prints the cost table.

Local equivalents live in `src/` (`python -m src.data.prepare`, `src.train.qlora`, ...).
`docker-compose.yml` runs vLLM + the FastAPI gateway on any machine with a GPU.

## Results (fill in after running)

| Variant | Size (GB) | Intent acc | Macro-F1 | JSON valid | p50 (s) | p95 (s) | tok/s @ c=32 | $/1M tokens |
|---|---|---|---|---|---|---|---|---|
| fp16 merged | | | | | | | | |
| AWQ int4 | | | | | | | | |
| GGUF Q4_K_M | | | | | | | | |
| Hosted API (baseline) | n/a | | | | | | n/a | |

## CI/CD

`.github/workflows/eval-gate.yml` evaluates the candidate GGUF on a CPU
runner against `configs/gate.yaml`. The publish job (gateway image to GHCR)
`needs` the gate, so a score drop blocks the release. Required secret:
`CANDIDATE_GGUF_URL` (e.g. the GGUF uploaded to the Hugging Face Hub).

## Notes

- Tune `configs/gate.yaml` floors after your first real eval run.
- None of this has been executed yet; expect to adjust library versions
  (TRL/AutoAWQ APIs move quickly).
