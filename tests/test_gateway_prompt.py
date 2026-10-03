"""The gateway's hard-coded prompt must equal what the training/eval Renderer produces."""
import pytest

from src.schema import SYSTEM_PROMPT

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
transformers = pytest.importorskip("transformers")


def test_gateway_prompt_matches_renderer():
    from src.formatting import build_renderer
    from src.serve.app import render_prompt

    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    except Exception as e:  # noqa: BLE001  (offline machine)
        pytest.skip(f"tokenizer not available: {e!r}"[:200])
    rend = build_renderer(tok)
    for msg in ["curly {braces} and {system}", "hello", "My card was stolen, what do I do?", 'quote " and newline\nhere']:
        want = rend.prompt([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": msg}])
        assert render_prompt(msg) == want
