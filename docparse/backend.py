"""Ollama (OpenAI-compatible) client for Xiaomi-OCR-0.

Payload and decoding parameters mirror the official MCP server exactly; using a
raw /chat/completions call without them degrades quality (the model card says so
explicitly).
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from . import imageops
from .config import BASE_URL, DECODE, MODEL


class BackendError(RuntimeError):
    pass


class ContextOverflow(BackendError):
    pass


def _check_loopback(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise BackendError("Model endpoint must be loopback (local inference only).")
    return url.rstrip("/")


def _post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        low = body.lower()
        if exc.code == 400 and ("maximum context" in low or "context length" in low):
            raise ContextOverflow(body[:800]) from exc
        raise BackendError(f"HTTP {exc.code}: {body[:800]}") from exc
    except urllib.error.URLError as exc:
        raise BackendError(
            f"Cannot reach {url}: {exc.reason}. Start the backend (see scripts/start_ollama.sh)."
        ) from exc


def ping(base_url: str = BASE_URL) -> bool:
    url = _check_loopback(base_url) + "/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def chat(
    image_png: bytes,
    prompt: str,
    max_tokens: int = 4096,
    *,
    presence_penalty: float | None = None,
    temperature: float | None = None,
    base_url: str = BASE_URL,
    model: str = MODEL,
    timeout: float = 600.0,
    retries: int = 1,
) -> dict[str, Any]:
    png = imageops.prepare(image_png)
    encoded = base64.b64encode(png).decode("ascii")
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
        **DECODE,
    }
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if temperature is not None:
        payload["temperature"] = temperature

    url = _check_loopback(base_url) + "/chat/completions"
    last: Exception | None = None
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            data = _post(url, payload, timeout)
        except ContextOverflow:
            raise
        except BackendError as exc:
            last = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        return {
            "text": str(content).strip(),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage") or {},
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }
    raise last or BackendError("unknown backend failure")
