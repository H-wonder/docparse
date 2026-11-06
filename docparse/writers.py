"""Renderers over the IR. Markdown is a *view*, not the source of truth."""

from __future__ import annotations

import json
from pathlib import Path

from .ir import Document, Page


def page_markdown(page: Page) -> str:
    synthetic = [b for b in page.blocks if b.type == "page_markdown"]
    if synthetic:
        return synthetic[0].content.strip()

    whole = [b for b in page.blocks if b.type == "page"]
    if whole:
        return whole[0].content.strip()

    parts: list[str] = []
    for b in sorted(page.blocks, key=lambda x: x.reading_order):
        if b.type == "figure":
            continue
        content = (b.html or b.content).strip()
        if content:
            parts.append(content)
    return "\n\n".join(parts)


def document_markdown(doc: Document) -> str:
    pages = [page_markdown(p) for p in doc.pages]
    pages = [p for p in pages if p]
    if len(pages) <= 1:
        return (pages[0] if pages else "") + "\n"
    chunks = []
    for i, text in enumerate(pages, 1):
        chunks.append(f"<!-- page {i} -->\n\n{text}")
    return "\n\n".join(chunks) + "\n"


def write(doc: Document, outdir: Path, *, write_json: bool = True) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(doc.path).stem
    paths: dict[str, Path] = {}

    md_path = outdir / f"{stem}.md"
    md_path.write_text(document_markdown(doc), encoding="utf-8")
    paths["markdown"] = md_path

    if write_json:
        json_path = outdir / f"{stem}.json"
        json_path.write_text(
            json.dumps(doc.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        paths["json"] = json_path
    return paths
