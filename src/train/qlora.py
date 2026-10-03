"""QLoRA fine-tune of a (multimodal) checkpoint on text-only chat data.

Prompts are pre-rendered with src.formatting (thinking off) and the loss is
computed on the completion only. Logs to MLflow; writes train_info.json.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Trainer, TrainingArguments

from src.config import load_cfg
from src.formatting import build_renderer, encode_rows
from src.modeling import compute_dtype, extra_input_keys, load_model


class Collator:
    def __init__(self, pad_id, extra_keys):
        self.pad_id, self.extra_keys = pad_id, extra_keys

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            lab[i, :k] = torch.tensor(b["labels"])
            att[i, :k] = 1
        out = {"input_ids": ids, "labels": lab, "attention_mask": att}
        for k in self.extra_keys:
            out[k] = torch.zeros_like(ids)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N train rows")
    ap.add_argument("--resume", action="store_true", help="continue from the latest checkpoint if one exists")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # torchrun starts one process per GPU. Keep the *effective* batch size equal to the
    # single-GPU recipe: per_device_batch x accum x world_size.
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    accum = cfg.get("gradient_accumulation_steps", 1)
    if accum % world:
        raise SystemExit(f"gradient_accumulation_steps={accum} not divisible by {world} GPUs")
    accum //= world
    main_proc = local_rank == 0
    if main_proc:
        print(f"world_size={world} per_device_batch={cfg['per_device_train_batch_size']} accum={accum} "
              f"-> effective batch {cfg['per_device_train_batch_size'] * accum * world}")

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    rend = build_renderer(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    train, d1 = encode_rows(cfg["train_file"], rend, tok, cfg["max_length"])
    val, d2 = encode_rows(cfg["val_file"], rend, tok, cfg["max_length"])
    if args.limit:
        train = train[: args.limit]
    print(f"train={len(train)} val={len(val)} dropped_too_long={d1 + d2}")

    model = load_model(cfg["base_model"], cfg["load_in_4bit"], cfg.get("loaders"))
    extra = extra_input_keys(model)
    print("extra text-only inputs:", extra)
    model.config.use_cache = False
    model.enable_input_require_grads()

    lc = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        target_modules=lc["target_regex"], task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_trainable == 0:
        raise SystemExit("LoRA matched no modules: fix lora.target_regex")

    # Picked up by transformers' MLflow callback (rank 0 only), so no rank touches the DB twice.
    os.environ["MLFLOW_TRACKING_URI"] = cfg["mlflow_tracking_uri"]
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg["mlflow_experiment"]
    save_steps = cfg.get("save_steps", 0)
    targs = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=accum,
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_ratio"],  # transformers 5.x: float in [0,1) = fraction of total steps
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="epoch",
        save_strategy="steps" if save_steps else "no",
        save_steps=save_steps or 500,
        save_total_limit=2,
        disable_tqdm=cfg.get("disable_tqdm", False),
        ddp_find_unused_parameters=False,
        bf16=compute_dtype() == torch.bfloat16,
        fp16=compute_dtype() == torch.float16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if cfg["load_in_4bit"] else "adamw_torch",
        seed=cfg["seed"],
        remove_unused_columns=False,
        report_to="mlflow",
        run_name=f"{cfg['name']}-train",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
                      data_collator=Collator(pad_id, extra))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    ckpts = sorted((out_dir / "trainer").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    resume = str(ckpts[-1]) if (args.resume and ckpts) else None
    if resume and main_proc:
        print("resuming from", resume)
    trainer.train(resume_from_checkpoint=resume)
    train_s = time.time() - t0

    if trainer.is_world_process_zero():
        model.save_pretrained(out_dir / "adapter")
        tok.save_pretrained(out_dir / "adapter")
    final_eval = trainer.evaluate()   # all ranks must take part
    info = {
        "name": cfg["name"], "base_model": cfg["base_model"],
        "train_examples": len(train), "dropped_too_long": d1 + d2,
        "trainable_params": n_trainable,
        "train_seconds": round(train_s, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "final_val_loss": final_eval.get("eval_loss"),
    }
    info.update(world_size=world, effective_batch=cfg["per_device_train_batch_size"] * accum * world)
    if main_proc:
        (out_dir / "train_info.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
