"""Render chat messages into (prompt, completion) strings for any model.

Why: Qwen3.5's template opens `<think>` by default and Gemma 4 has its own
thinking/turn tokens. If training and inference render the prompt differently,
quality collapses silently. So we render ONCE, with thinking off, and use the
same strings for training, eval and (later) serving.

The completion's end-of-turn token is discovered by rendering a full
conversation and reading what the template puts after the assistant text, so
nothing is hard-coded per model.
"""
import json
from dataclasses import dataclass

_PROBE = "PROBE_ANSWER_TEXT"


@dataclass
class Renderer:
    tok: object
    end_str: str          # text after the answer that ends the turn, e.g. "<|im_end|>"
    answer_prefix: str    # text the template inserts between prompt and answer (usually "")
    end_ids: list         # token ids that should stop generation

    def prompt(self, messages) -> str:
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def completion(self, answer: str) -> str:
        return self.answer_prefix + answer + self.end_str


def build_renderer(tok) -> Renderer:
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    prompt = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full = tok.apply_chat_template(
        msgs + [{"role": "assistant", "content": _PROBE}],
        tokenize=False, enable_thinking=False,
    )
    i = full.find(_PROBE)
    if i < 0:
        raise ValueError("assistant text not found in rendered template")
    end_str = full[i + len(_PROBE):].rstrip()
    # Anything the template puts between the generation prompt and the answer
    # (only if the full render shares the prompt as a prefix).
    answer_prefix = ""
    if full.startswith(prompt):
        answer_prefix = full[len(prompt):i]
    ids = []
    for s in {end_str, tok.eos_token}:
        if not s:
            continue
        tid = tok.convert_tokens_to_ids(s)
        if tid is not None and tid != tok.unk_token_id:
            ids.append(tid)
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    return Renderer(tok, end_str, answer_prefix, sorted(set(ids)))


def encode_rows(path, rend, tok, max_len):
    feats, dropped = [], 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        p = tok(rend.prompt(r["prompt"]), add_special_tokens=False)["input_ids"]
        c = tok(rend.completion(r["completion"][0]["content"]), add_special_tokens=False)["input_ids"]
        if len(p) + len(c) > max_len:
            dropped += 1
            continue
        feats.append({"input_ids": p + c, "labels": [-100] * len(p) + c})
    return feats, dropped
