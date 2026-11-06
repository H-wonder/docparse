"""Automatic quality gates.

The goal is to replace human review: every failure mode the pipeline can detect
must either (a) trigger an automatic retry/escalation, or (b) be recorded as a
machine-readable warning with a confidence score.
"""

from __future__ import annotations

import re
from typing import Any

from postprocess.document import finalize_document
from postprocess.otsl import convert_otsl_to_html
from postprocess.repetition_guard import analyze_repetition

OTSL_RE = re.compile(r"<(?:fcel|ecel|nl|lcel|ucel|xcel)>")


def warning(code: str, severity: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, **extra}


def finalize(text: str) -> str:
    """Upstream whole-page cleanup: repetition truncation, math delimiters, OTSL→HTML."""
    try:
        return finalize_document(text)
    except Exception:
        return text


def convert_table(otsl: str) -> tuple[str, bool]:
    """Return (html, ok). Never raise: raw OTSL is kept when conversion fails."""
    try:
        html = convert_otsl_to_html(otsl)
        return (html if html else otsl), bool(html)
    except Exception:
        return otsl, False


def assess(
    text: str,
    finish_reason: str | None = None,
    *,
    mode: str = "page",
    min_chars: int = 1,
) -> dict[str, Any]:
    """Score one OCR response and decide whether an automatic retry is warranted."""
    text = text or ""
    warnings: list[dict[str, Any]] = []
    score = 1.0

    if len(text.strip()) < min_chars:
        warnings.append(warning("empty_output", "high",
                                "Model returned no usable text for this region/page."))
        score = 0.0

    analysis = analyze_repetition(text, finish_reason)
    if analysis["suspicious"]:
        warnings.append(warning("repetition", "medium",
                                "Repetitive output detected; an automatic retry will be attempted.",
                                reasons=analysis["reasons"]))
        score -= 0.5

    if finish_reason == "length":
        warnings.append(warning("truncated", "high",
                                "Output hit the token budget and was truncated."))
        score -= 0.4

    if OTSL_RE.search(text):
        _, ok = convert_table(text)
        if not ok:
            warnings.append(warning("table_parse_failed", "medium",
                                    "Table output could not be converted; raw OTSL kept."))
            score -= 0.2

    score = max(0.0, min(1.0, score))
    return {
        "warnings": warnings,
        "score": round(score, 3),
        "suspicious": bool(analysis["suspicious"]),
        "needs_retry": bool(analysis["suspicious"]) or finish_reason == "length"
        or len(text.strip()) < min_chars,
        "analysis": analysis,
        "mode": mode,
    }


def ink_ratio(image) -> float:
    """Fraction of dark pixels: a cheap proxy for how much content a page holds."""
    try:
        import numpy as np
        arr = np.asarray(image.convert("L"), dtype=np.uint8)
        return float((arr < 200).mean())
    except Exception:
        return 0.0


def coverage_warning(text: str, image) -> dict | None:
    """Catch the failure mode where the model returns only a title or a line."""
    ink = ink_ratio(image)
    chars = len(re.sub(r"\s", "", text or ""))
    if ink > 0.03 and chars < 30:
        return warning("low_coverage", "high",
                       f"Page looks dense (ink {ink:.1%}) but output has only {chars} chars.",
                       ink=round(ink, 4), chars=chars)
    if ink > 0.06 and chars < 100:
        return warning("low_coverage", "medium",
                       f"Page looks dense (ink {ink:.1%}) but output has only {chars} chars.",
                       ink=round(ink, 4), chars=chars)
    return None


def parse_kie(raw: str, fields: list[str] | None = None) -> tuple[dict | None, str]:
    """Extract the first JSON object from a KIE response."""
    import json
    match = re.search(r"\{[\s\S]*\}", raw or "")
    if not match:
        return None, "no_json_object"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(data, dict):
        return None, "not_an_object"
    if fields:
        data = {f: data.get(f, "") for f in fields}
    return data, ""


def _digits(value: str) -> str:
    return re.sub(r"[^\d]", "", value or "")


def ground_fields(data: dict, page_text: str) -> dict[str, dict]:
    """Hallucination check: every extracted value must appear in the OCR text.

    Numeric values are compared digit-by-digit (so 1,234.56 matches 1234.56 and
    2024-01-05 matches 2024年1月5日); text values are compared after whitespace
    stripping.
    """
    text_digits = _digits(page_text)
    text_flat = re.sub(r"\s+", "", page_text or "")
    results: dict[str, dict] = {}
    for key, value in (data or {}).items():
        if not isinstance(value, str) or not value.strip():
            results[key] = {"ok": False, "reason": "empty"}
            continue
        digits = _digits(value)
        if len(digits) >= 3:
            ok = digits in text_digits
        else:
            ok = re.sub(r"\s+", "", value) in text_flat
        results[key] = {"ok": ok, "reason": "" if ok else "not_found_in_ocr_text"}
    return results


NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def numeric_tokens(text: str) -> list[str]:
    return NUM_RE.findall(text or "")


def cross_check_numbers(primary: str, secondary: str, *, limit: int = 40) -> dict[str, Any]:
    """Compare numeric tokens between two independent readings.

    Used by the automatic second-pass verifier for money/date/id-heavy documents.
    Returns the tokens that disagree; an empty list means agreement.
    """
    a = numeric_tokens(primary)[:limit]
    b = numeric_tokens(secondary)[:limit]
    if not a and not b:
        return {"ok": True, "mismatches": [], "compared": 0}
    mismatches = []
    for i, (x, y) in enumerate(zip(a, b)):
        if x.replace(",", "") != y.replace(",", ""):
            mismatches.append({"index": i, "primary": x, "secondary": y})
    if len(a) != len(b):
        mismatches.append({"index": min(len(a), len(b)),
                           "primary": f"{len(a)} tokens", "secondary": f"{len(b)} tokens"})
    total = max(len(a), len(b), 1)
    return {"ok": not mismatches, "mismatches": mismatches[:10],
            "compared": total, "agreement": round(1 - len(mismatches) / total, 3)}
