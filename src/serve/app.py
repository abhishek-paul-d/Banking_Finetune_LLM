"""FastAPI gateway in front of vLLM / llama.cpp: input validation and output schema enforcement.

The model was trained on prompts rendered with thinking OFF, so the gateway sends that exact prompt text to the
raw /v1/completions endpoint (not /v1/chat/completions, whose default template for Qwen3.5 turns thinking ON).
tests/test_gateway_prompt.py checks this template against the real tokenizer so the two cannot drift apart.
"""
import logging
import os
import time

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.schema import SYSTEM_PROMPT, parse_output

VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "banking-triage")

# Qwen3.5 chat template with enable_thinking=False (verified against the tokenizer in the tests).
PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def render_prompt(message: str) -> str:
    # str.replace, not str.format: user text may contain braces
    return PROMPT_TEMPLATE.replace("{system}", SYSTEM_PROMPT).replace("{message}", message)


log = logging.getLogger("triage")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Banking Triage")
client = httpx.AsyncClient(timeout=60)


class TriageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class TriageResponse(BaseModel):
    intent: str
    urgency: str
    needs_human: bool


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/triage", response_model=TriageResponse)
async def triage(req: TriageRequest):
    t0 = time.perf_counter()
    payload = {
        "model": MODEL_NAME,
        "temperature": 0,
        "max_tokens": 64,
        "stop": ["<|im_end|>"],
        "prompt": render_prompt(req.message),
    }
    try:
        r = await client.post(f"{VLLM_URL}/v1/completions", json=payload)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"model backend error: {e}")

    parsed = parse_output(r.json()["choices"][0]["text"])
    log.info("triage latency=%.3fs valid=%s", time.perf_counter() - t0, parsed is not None)
    if parsed is None:
        raise HTTPException(502, "model returned invalid JSON")
    return parsed
