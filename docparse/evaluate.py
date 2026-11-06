"""Compare parse output against reference Markdown.

This is deliberately model-free: it lets you A/B page vs region mode, DPI, and
backend versions on *your* documents instead of trusting benchmark tables.
"""

from __future__ import annotations

import html as _html
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from .pipeline import Parser
from .writers import document_markdown


def normalize(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text or "", flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = _html.unescape(text)
    text = re.sub(r"\s+", "", text)
    return text


def compare(reference: str, hypothesis: str) -> dict:
    ref, hyp = normalize(reference), normalize(hypothesis)
    matcher = SequenceMatcher(None, ref, hyp, autojunk=False)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    distance = len(ref) + len(hyp) - 2 * matches
    return {
        "similarity": round(matcher.ratio(), 4),
        "cer": round(distance / max(1, len(ref)), 4),
        "ref_chars": len(ref),
        "hyp_chars": len(hyp),
        "length_ratio": round(len(hyp) / max(1, len(ref)), 3),
    }


def evaluate(inputs: list[Path], refs_dir: Path, parser: Parser) -> dict:
    rows = []
    for path in inputs:
        ref_path = refs_dir / f"{path.stem}.md"
        if not ref_path.exists():
            rows.append({"input": str(path), "status": "missing_reference"})
            continue
        try:
            doc = parser.parse(path)
            hypothesis = document_markdown(doc)
            row = compare(ref_path.read_text(encoding="utf-8"), hypothesis)
        except Exception as exc:
            rows.append({"input": str(path), "status": "failed", "error": str(exc)})
            continue
        row.update({
            "input": str(path),
            "status": "ok",
            "mode": ",".join(sorted({p.mode for p in doc.pages if p.mode})),
            "needs_review": doc.needs_review,
            "warnings": [w for w in doc.warnings if w.get("severity") in {"high", "medium"}],
        })
        rows.append(row)
    scored = [r for r in rows if r["status"] == "ok"]
    summary = {
        "documents": len(rows),
        "scored": len(scored),
        "mean_similarity": round(sum(r["similarity"] for r in scored) / len(scored), 4) if scored else None,
        "mean_cer": round(sum(r["cer"] for r in scored) / len(scored), 4) if scored else None,
        "needs_review": sum(1 for r in scored if r["needs_review"]),
    }
    return {"summary": summary, "rows": rows}


def write_report(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
