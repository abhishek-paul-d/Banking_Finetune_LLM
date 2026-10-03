"""Model loading and small compatibility helpers shared by training and eval."""
import inspect
import os


def compute_dtype():
    """bf16 on Ampere+ (A100, L4...), fp16 on T4/P100 (no native bf16)."""
    import torch

    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


def load_model(base_model: str, load_in_4bit: bool = True, loaders=None, adapter: str | None = None,
               decompress: bool = False):
    """Load a (possibly multimodal) checkpoint for text-only use on GPU 0.

    Qwen3.5 and Gemma 4 ship as multimodal checkpoints (weights under
    `model.language_model.*`), so we try the image-text-to-text auto class
    first and fall back to the causal-LM one.
    """
    import torch
    import transformers
    from transformers import BitsAndBytesConfig

    dt = compute_dtype()
    print("compute dtype:", dt)
    dev = int(os.environ.get("LOCAL_RANK", 0))   # one process per GPU under torchrun
    kw = dict(dtype=dt, device_map={"": dev})
    if load_in_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dt,
        )
    if decompress:
        # compressed-tensors (llm-compressor) checkpoints: expand the 4-bit weights back to fp16 so we can
        # measure the *quality* of the quantized weights with plain transformers. Speed is benchmarked in vLLM.
        from transformers import CompressedTensorsConfig

        kw["quantization_config"] = CompressedTensorsConfig(run_compressed=False)
    last = None
    for name in loaders or ["AutoModelForImageTextToText", "AutoModelForCausalLM"]:
        try:
            model = getattr(transformers, name).from_pretrained(base_model, **kw)
            print(f"loaded {base_model} via {name}: {type(model).__name__}")
            break
        except Exception as e:  # noqa: BLE001
            print(f"{name} failed: {type(e).__name__}: {str(e)[:200]}")
            last = e
    else:
        raise last
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model


def extra_input_keys(model) -> list[str]:
    """Text-only inputs some multimodal models still require (all-zeros)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    params = inspect.signature(base.forward).parameters
    return [k for k in ("token_type_ids", "mm_token_type_ids") if k in params]
