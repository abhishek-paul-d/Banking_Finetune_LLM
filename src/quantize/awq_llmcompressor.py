"""AWQ 4-bit (W4A16) quantization of the merged model with llm-compressor.

AutoAWQ is archived and predates Qwen3.5, so we use llm-compressor, which has Qwen3.5 AWQ mappings.
Output is the "compressed-tensors" format that vLLM can load.

Two variants, because Qwen3.5 is a hybrid model with a new kind of attention layer ("linear_attn",
Gated DeltaNet) that is known to be touchy to quantize:
  default   : quantize everything in the text decoder EXCEPT linear_attn
              (this mirrors Red Hat's official Qwen3.5-4B W4A16 build)
  mlp_only  : additionally keep ALL attention layers in fp16, quantizing only the MLPs
              (more conservative, bigger file)
The vision tower, lm_head and embeddings are never quantized.

Calibration data: our own rendered training prompts + answers, so the quantizer sees the real task.
"""
import argparse
import json

import torch
from datasets import Dataset
from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

BASE_IGNORE = ["lm_head", "re:.*embed_tokens$", "re:.*visual.*", "re:.*mtp.*", "re:.*linear_attn.*"]
VARIANTS = {
    "default": BASE_IGNORE,
    "mlp_only": BASE_IGNORE + ["re:.*self_attn.*"],
}


def calibration_dataset(path, tok, n, max_len):
    from src.formatting import build_renderer

    rend = build_renderer(tok)
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(rend.prompt(r["prompt"]) + rend.completion(r["completion"][0]["content"]))
            if len(texts) >= n:
                break
    ds = Dataset.from_dict({"text": texts})
    return ds.map(
        lambda s: tok(s["text"], truncation=True, max_length=max_len,
                      add_special_tokens=False, return_attention_mask=True),
        remove_columns=["text"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/merged-fp16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=list(VARIANTS), required=True)
    ap.add_argument("--calib", default="data/train.jsonl")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        processor = AutoProcessor.from_pretrained(args.model)   # needs preprocessor_config.json in the folder
    except Exception as e:  # noqa: BLE001
        print("could not load processor, falling back to the tokenizer:", type(e).__name__)
        processor = None
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype=torch.float16)
    ds = calibration_dataset(args.calib, tok, args.samples, args.max_len)

    recipe = [
        AWQModifier(),
        QuantizationModifier(ignore=VARIANTS[args.variant], scheme="W4A16", targets=["Linear"]),
    ]
    extra = {"processor": processor} if processor is not None else {}
    oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=args.max_len,
            num_calibration_samples=len(ds), pad_to_max_length=False, **extra)

    model.save_pretrained(args.out, save_compressed=True)
    tok.save_pretrained(args.out)
    if processor is not None:
        processor.save_pretrained(args.out)   # vLLM needs these files to load a Qwen3.5 folder
    print(f"AWQ ({args.variant}) -> {args.out}")


if __name__ == "__main__":
    main()
