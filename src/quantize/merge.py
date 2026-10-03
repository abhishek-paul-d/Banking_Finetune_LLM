"""Merge the trained LoRA adapter into the full-precision base model.

After this, the model is one ordinary fp16 checkpoint (no adapter needed), which is
the starting point for every quantized variant (GGUF, AWQ).

Note: the adapter was trained on a 4-bit copy of the base model but is merged into the
fp16 base, which is the standard approach. It can cause tiny drift, so the notebook
evaluates the merged model before any quantization.
"""
import argparse

import torch
import transformers
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", default="outputs/full/qwen3_5_4b/adapter")
    ap.add_argument("--out", default="artifacts/merged-fp16")
    args = ap.parse_args()

    # Same class used for training (the checkpoint is a multimodal wrapper), so adapter keys line up.
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.float16, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.save_pretrained(args.out, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.out)
    # Qwen3.5 is a vision-language checkpoint: llm-compressor and vLLM both need preprocessor_config.json in the folder.
    AutoProcessor.from_pretrained(args.base).save_pretrained(args.out)
    print(f"merged model ({type(model).__name__}) -> {args.out}")


if __name__ == "__main__":
    main()
