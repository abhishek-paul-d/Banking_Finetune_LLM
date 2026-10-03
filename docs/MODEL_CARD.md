---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
datasets:
  - PolyAI/banking77
language:
  - en
tags:
  - text-classification
  - json
  - banking
  - qlora
  - gguf
  - awq
---

# Banking triage, fine-tuned Qwen3.5-4B (fp16, AWQ, GGUF)

Reads one customer-support message and returns a single JSON object:

```json
{"intent": "card_arrival", "urgency": "low", "needs_human": false}
```

`intent` is one of the 77 [Banking77](https://huggingface.co/datasets/PolyAI/banking77) labels. `urgency` and `needs_human` are **rule-derived from the intent**
(not human labels), so treat them as a convenience, not an independent judgment.

## Files

| Path | What | Use with |
|---|---|---|
| `merged-fp16/` | LoRA merged into fp16 base | vLLM, transformers |
| `awq-default/`, `awq-mlp-only/` | AWQ 4-bit (compressed-tensors), two layer-selection variants | vLLM |
| `gguf/model-Q8_0.gguf`, `gguf/model-Q4_K_M.gguf` | text-only GGUF | llama.cpp (converted with `--no-mtp`) |

## Quality (3,080 held-out test examples, scored through the running server)

| variant | intent accuracy | macro-F1 | valid JSON |
|---|---|---|---|
| fp16 | 0.9373 | 0.9375 | 100% |
| GGUF Q8_0 | 0.9370 | 0.9372 | 100% |
| GGUF Q4_K_M | 0.9357 | 0.9362 | 100% |
| AWQ mlp-only | 0.9331 | 0.9333 | 100% |
| AWQ default | 0.9315 | 0.9317 | 100% |

The base model without fine-tuning scores about 0.63 intent accuracy on this task.

## How to prompt it (important)

The model was trained on prompts rendered **with thinking disabled**. Send exactly this text to a raw completion endpoint (not the chat endpoint, whose default template for Qwen3.5 enables thinking):

```
<|im_start|>system
You are a banking support triage assistant. Read the customer message and reply with ONLY a JSON object with keys: "intent" (snake_case label), "urgency" (low|medium|high), "needs_human" (true|false).<|im_end|>
<|im_start|>user
{message}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Use temperature 0 and stop on `<|im_end|>`. A different prompt format can silently degrade quality.

## Training

QLoRA on 9,003 Banking77 training examples (4-bit NF4 base, LoRA rank 8, alpha 16, 2 epochs, effective batch 16), loss on the completion only.
Details, comparisons and benchmarks: see the project repository (`README.md`, `docs/FINDINGS.md`).

## Limitations

- Banking77 is a clean, single-domain, English dataset; real customer messages are messier and may fall outside the 77 intents. There is no "other" class, so the model will always pick one.
- Some intents overlap in meaning, which limits attainable accuracy; about 6% of test messages are still misclassified.
- The 4-bit variants lose up to 0.6 points of accuracy against fp16 (AWQ) or are within noise (GGUF).
- Not evaluated for fairness, adversarial input, or non-English text. Do not use it to make decisions about customers without human review.
