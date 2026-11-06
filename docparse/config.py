"""Runtime configuration. Everything stays on loopback by design."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ── model backend ────────────────────────────────────────────────────────────
MODEL = os.getenv("XIAOMI_OCR_MODEL", "longwayxu/xiaomi-ocr-0:bf16")
BASE_URL = os.getenv("XIAOMI_OCR_LOCAL_URL", "http://127.0.0.1:11434/v1").rstrip("/")

# The model's documented image bound. Bigger inputs are downscaled by upstream.
MAX_EDGE = 2048
# Total-pixel bounds documented in OLLAMA.md (16px patches, 2x merge).
MIN_PIXELS = 65_536
MAX_PIXELS = 16_777_216

# ── prompts (verbatim from the official pipeline) ────────────────────────────
PROMPTS = {
    "document": (
        "Extract all information from the main body of the document image and represent it "
        "in markdown format, ignoring headers and footers. Tables should be expressed in OTSL "
        "format, formulas in the document should be represented using LATEX format, and the "
        "parsing should be organized according to the reading order."
    ),
    "text": "Extract the text in the image.",
    "table": "Parse the table in the image into OTSL.",
    "formula": "Identify the formula in the image and represent it using LATEX format.",
}

REGION_MAX_TOKENS = {"title": 256, "text": 1024, "table": 2048, "formula": 512, "figure": 256}


def kie_prompt(fields: list[str]) -> str:
    """Schema-constrained extraction prompt (mirrors the official MCP server)."""
    import json
    schema = {str(field): "" for field in fields}
    return (
        "Extract key information in the image\n\n"
        "Please output the key information in JSON format according to the following schema:\n"
        + json.dumps(schema, ensure_ascii=False, indent=4)
    )

# ── decoding (aligned with the official MCP server) ───────────────────────────
DECODE = {
    "temperature": 0.0,
    "top_p": 0.8,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "chat_template_kwargs": {"enable_thinking": False},
}
# Used only for the automatic repetition-retry pass.
RETRY_PRESENCE_PENALTY = 1.5


@dataclass
class ParseOptions:
    dpi: int = 150
    max_pages: int | None = None
    mode: str = "auto"                 # auto | page | region
    max_tokens: int = 8192
    retry_repetition: bool = True
    layout_device: str = "cpu"         # keep CPU: the 8GB GPU belongs to the VLM
    save_page_images: bool = False
    outdir: Path | None = None
    verbose: bool = False
    kie_fields: list[str] | None = None


# Layout label → task bucket. Mirrors evaluation_pipeline/twostage_img2md.py.
LABEL_TASK = {
    "text": ["abstract", "algorithm", "content", "doc_title", "figure_title",
             "paragraph_title", "reference_content", "text", "vertical_text",
             "vision_footnote", "seal", "formula_number", "header", "footer",
             "number", "footnote", "aside_text", "reference"],
    "table": ["table"],
    "formula": ["display_formula", "inline_formula"],
    "skip": ["chart", "image", "footer_image", "header_image"],
}

SKIP_LABELS = {"header", "footer", "header_image", "footer_image"}
