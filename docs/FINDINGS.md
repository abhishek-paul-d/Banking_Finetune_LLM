# Findings and write-up

Everything below is backed by a file in `results/` (names in brackets). Where something is an inference rather than a measurement, it says so.

## 1. The task and the data

Banking77 has 13,083 customer messages across 77 intents. We split it into 9,003 train, 1,000 validation and 3,080 test examples (stratified, with a
leakage check) and added two rule-derived fields, `urgency` and `needs_human`, from the intent (`src/schema.py`). The model must output one JSON object.
Because those two fields are deterministic functions of the intent, **the real difficulty is intent classification**; urgency and needs_human accuracy
(about 98% and 99%) follow from it.

## 2. Picking the model (Phase 1) [`results/pilot/comparison.md`]

Qwen3.5-4B and Gemma-4-E4B were fine-tuned identically on a 2,000-example pilot set and scored on the full test set.

| | Qwen3.5-4B | Gemma-4-E4B |
|---|---|---|
| zero-shot intent accuracy (770 ex.) | 0.630 | 0.665 |
| fine-tuned intent accuracy | 0.898 | 0.896 |
| fine-tuned macro-F1 | 0.903 | 0.900 |
| train time | 3,906 s | 2,341 s |
| train peak VRAM | 5.8 GB | 12.4 GB |
| eval tokens/s | 68 | 58 |

The quality difference was +0.25 points (95% paired-bootstrap CI -0.55 to +1.15): **a tie**. The pre-registered rule said to prefer the faster/smaller model; Qwen3.5-4B
uses less than half the memory and generates faster, so it won. Gemma trained faster in this setup, so "faster" here means inference speed and memory. The plan said to rerun with a second
seed when close; **that was not done**. The choice is a defensible tie-break, not a proven superiority.

## 3. The fine-tune (Phase 2) [`results/phase2/`]

QLoRA (4-bit base, LoRA rank 8, alpha 16, 15.2M trainable parameters, effective batch 16, 2 epochs) on all 9,003 training examples, trained across two T4 GPUs in 6,348 s.
Rank was lowered from the pilot's 16 to 8 because the task is narrow; the full run reached **0.9393 intent accuracy, 0.9398 macro-F1, 100% valid JSON, 0.13% invented labels** on the test set.
(The pilot's rank-16 and the full run's rank-8 were not compared head to head.)

Two evaluation details that matter: prompts are rendered once with thinking disabled and reused for training, scoring and serving, since a template mismatch silently destroys quality in this model family.
Macro-F1 is computed over the 77 real labels only; an earlier version also counted invented labels as extra classes, which understated the score.

## 4. Compression (Phase 3) [`results/phase3/quant_table_all.md`]

| variant | size (GB) | intent acc | vs fp16 | same answer as fp16 |
|---|---|---|---|---|
| fp16 merged | 9.10 | 0.9373 | | |
| GGUF Q8_0 | 4.48 | 0.9383 | +0.10 pt | 99.8% |
| GGUF Q4_K_M | 2.71 | 0.9357 | -0.16 pt | 98.7% |
| AWQ mlp_only | 5.80 | 0.9331 | -0.42 pt | 98.3% |
| AWQ default | 5.37 | 0.9312 | -0.62 pt | 97.9% |

- Merging the adapter into the fp16 base did no visible damage: merged fp16 scored 0.9373 against 0.9393 for the adapter on the 4-bit base, a 0.2 point (6 example) difference.
- Q8_0 and Q4_K_M are indistinguishable from fp16 under a paired McNemar test (p = 0.25, 0.44). AWQ is measurably but slightly worse (p = 0.04 for mlp_only, p < 0.01 for default).
- AWQ is the *worse* compression here: lower quality than Q4_K_M and a larger file. Qwen3.5 has new `linear_attn` layers, big embeddings and a vision tower that we deliberately kept in 16-bit
  (quantizing `linear_attn` is known to be fragile), so only the MLP and standard attention shrink. AutoAWQ is archived and predates Qwen3.5, so llm-compressor (0.14.0) was used.

## 5. Serving (Phase 4) [`results/phase4/`]

Setup: one NVIDIA L4, prompts identical across servers (raw `/v1/completions`, same rendered text), temperature 0, 64 max tokens, concurrency levels 1 to 128, at least 300 requests per level.
Accuracy re-measured through each server matched Phase 3 within 0.2 points for every variant, so the serving stack does not change the model's answers.

Headline numbers are in the README table. Points worth stating carefully:

- **vLLM vs llama.cpp is a comparison of whole stacks**, not just file formats: the GGUF rows differ in both format and server.
- **Latency vs throughput trade.** llama.cpp Q4_K_M has the lowest single-request p50 (468 ms) and the fastest start (12 s). vLLM wins everywhere once several requests arrive at once.
- **vLLM AWQ is the cheapest service**: about 27% higher peak throughput than fp16 for a 0.4 to 0.6 point accuracy cost. Whether that trade is worth it depends on how much one point of accuracy is worth to you.
- **Cost** is GPU time only: `hourly / (req_per_s x 3600)`, at the best load level (GPU always busy). With Colab's L4 at 1.56 compute units/hour and $9.99 per 100 units this is $0.156/h
  (check your own plan's price). Gateway, networking, storage, idle time and engineering are not included. [`results/phase4/cost_table.md`]

### Why llama.cpp scales badly on this model [`results/phase4/tuning_summary.csv`]

In the Phase 4 server log, 28 or more request slots were busy at the same moment, so requests were not queueing. Yet each request got one token every 190 to 260 ms under load, against about 21 ms alone:
decoding steps slowed almost in proportion to the batch, so overall throughput grew about 2x instead of 15x (vLLM).

A follow-up on the same L4 varied the setup (output tokens/s at 1 / 8 / 32 simultaneous requests):

| setting | 1 | 8 | 32 | scaling |
|---|---|---|---|---|
| 64 slots (Phase 4 setting) | 51.7 | 101.6 | 106.5 | 2.1x |
| 16 slots | 51.9 | 102.1 | 107.9 | 2.1x |
| 8 slots | 51.2 | 100.7 | 106.2 | 2.1x |
| 1 slot | 56.9 | 57.5 | 57.5 | 1.0x |
| 16 slots, flash attention off | 45.9 | 95.1 | 91.0 | 2.0x |
| 16 slots, larger prompt batch | 51.1 | 100.4 | 106.7 | 2.1x |
| control: plain-attention Qwen3-4B, 64 slots | 73.8 | 262.1 | 471.0 | 6.4x |

No setting helps; the control model scales about three times better under the same settings. **Inference, not proof:** this points at the Gated-DeltaNet layers of Qwen3.5 being handled less efficiently in batches
by llama.cpp. The control differs from our model in several ways (size, source of the GGUF, no fine-tuning, shorter generation cap), and we did not profile the layers. The conclusion is limited to
"llama.cpp, at the version we built, with this model".

## 6. The CI gate (Phase 5)

A fixed 200-example slice (balanced over 77 intents) is scored by the Q4_K_M GGUF running on a free GitHub CPU runner. Thresholds (`configs/gate.yaml`) sit about four standard errors below the measured
score. Unit tests show that good predictions pass and that prose output, 25% wrong labels, invented labels and missing metrics all fail. Both outcomes were demonstrated in GitHub Actions on the same slice and CPU runner:
the fine-tuned Q4_K_M **passed** (intent accuracy 0.9600, macro-F1 0.9553, 100% valid JSON, 0.5% invented labels, urgency 0.995, needs_human 1.000), and the un-tuned Qwen3.5-4B Q4_K_M
(`unsloth/Qwen3.5-4B-GGUF`) **failed** on 5 of 6 checks (intent accuracy 0.025, macro-F1 0.033, urgency 0.635, needs_human 0.795, invented labels 97.5%).
The evaluation step took 22 min 30 s on the free CPU runner. Two observations: the CPU runner reproduced the GPU scores exactly, and the un-tuned model passed the JSON-validity check (100%) while inventing intent names, so a validity check alone would not have caught it.
The gate detects broken models, not subtle regressions (noise of about 1.4 points on 200 examples).

## 7. Things that went wrong (and what they taught)

- The Banking77 HF repo is script-based and newer `datasets` refuses it; we loaded the same CSVs from the original GitHub source.
- macro-F1 was initially understated because invented labels counted as classes; fixed and recomputed everywhere.
- Colab/Kaggle sessions erase outputs when they end; every phase now saves and downloads results immediately.
- llama.cpp could not load the first GGUF (the multi-token-prediction head was included); converting with `--no-mtp` fixed it.
- The first gateway version used the chat endpoint, which turns thinking on for Qwen3.5 and would have silently mismatched the training prompt; it now sends the training prompt text, with a test.

## 8. What would make this stronger

A second seed for the model comparison; a hosted-API cost baseline; constrained decoding to one of the 77 labels (would remove the 0.1% invented labels); testing on messier real-world text;
running the gateway under load with a real client mix; and a cloud GPU price in place of the Colab-based one.
