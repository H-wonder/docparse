"""PDF probing, native text extraction, and page rendering (PyMuPDF)."""

from __future__ import annotations

import hashlib
import html as _html
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from .ir import Block
from .config import SKIP_LABELS

BAD_CHARS = set("�\u0000")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f�]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class PageProbe:
    number: int
    width: float
    height: float
    chars: int
    bad_ratio: float
    image_coverage: float
    kind: str = "native"          # native | scanned | hybrid | blank
    quality: float = 1.0


def probe_page(page: fitz.Page, number: int) -> PageProbe:
    text = page.get_text("text") or ""
    stripped = text.strip()
    chars = len(stripped)
    bad = len(_CTRL_RE.findall(stripped))
    bad_ratio = bad / max(1, chars)

    area = max(1.0, page.rect.width * page.rect.height)
    covered = 0.0
    try:
        for info in page.get_image_info():
            r = fitz.Rect(info["bbox"]) & page.rect
            if not r.is_empty:
                covered += r.get_area()
    except Exception:
        pass
    coverage = min(1.0, covered / area)

    if chars < 10:
        kind = "blank" if coverage < 0.05 else "scanned"
    elif bad_ratio > 0.05:
        kind = "scanned"                 # broken text layer → OCR is safer
    elif coverage > 0.6 and chars < 50:
        kind = "scanned"
    elif coverage > 0.3:
        kind = "hybrid"
    else:
        kind = "native"

    quality = 1.0
    if kind in {"scanned", "blank"}:
        quality = 0.0
    else:
        quality = max(0.0, 1.0 - bad_ratio * 10)
        if chars < 80 and coverage < 0.3:
            quality *= 0.6
    return PageProbe(number, page.rect.width, page.rect.height,
                     chars, bad_ratio, coverage, kind, quality)


def probe_pdf(path: Path) -> list[PageProbe]:
    with fitz.open(path) as doc:
        return [probe_page(doc[i], i + 1) for i in range(len(doc))]


# ── native extraction ────────────────────────────────────────────────────────

def _norm(bbox, w: float, h: float) -> list[float]:
    return [round(bbox[0] / w * 1000, 1), round(bbox[1] / h * 1000, 1),
            round(bbox[2] / w * 1000, 1), round(bbox[3] / h * 1000, 1)]


def _rows_to_html(rows: list[list[str | None]]) -> str:
    out = ["<table>"]
    for row in rows:
        out.append("<tr>" + "".join(
            f"<td>{_html.escape(c or '').strip()}</td>" for c in row) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def _two_column(blocks: list[dict], width: float) -> bool:
    if len(blocks) < 6:
        return False
    left = sum(1 for b in blocks if (b["bbox"][0] + b["bbox"][2]) / 2 < width * 0.5)
    right = len(blocks) - left
    if left < len(blocks) * 0.3 or right < len(blocks) * 0.3:
        return False
    # A block straddling the gutter means single-column flow.
    straddle = sum(1 for b in blocks
                   if b["bbox"][0] < width * 0.4 and b["bbox"][2] > width * 0.6)
    return straddle <= max(1, len(blocks) * 0.1)


def _reading_order(blocks: list[dict], width: float, height: float) -> list[dict]:
    if _two_column(blocks, width):
        left = sorted([b for b in blocks if (b["bbox"][0] + b["bbox"][2]) / 2 < width * 0.5],
                      key=lambda b: (b["bbox"][1], b["bbox"][0]))
        right = sorted([b for b in blocks if (b["bbox"][0] + b["bbox"][2]) / 2 >= width * 0.5],
                       key=lambda b: (b["bbox"][1], b["bbox"][0]))
        return left + right
    return sorted(blocks, key=lambda b: (round(b["bbox"][1] / 8), b["bbox"][0]))


def extract_native(page: fitz.Page, page_no: int) -> list[Block]:
    """Vector text + ruled tables + image regions, in reading order."""
    w, h = page.rect.width, page.rect.height
    raw_blocks: list[dict] = []
    sizes: list[float] = []

    data = page.get_text("dict")
    for b in data.get("blocks", []):
        if b.get("type") != 0:
            continue
        spans = [s for line in b.get("lines", []) for s in line.get("spans", [])]
        text = "".join(s.get("text", "") for s in spans).strip()
        if not text:
            continue
        size = max((s.get("size", 0) for s in spans), default=0)
        sizes.append(size)
        raw_blocks.append({"bbox": list(b["bbox"]), "text": text, "size": size})

    table_rects: list[fitz.Rect] = []
    table_blocks: list[dict] = []
    if hasattr(page, "find_tables"):
        try:
            for table in page.find_tables().tables:
                rows = table.extract()
                if not rows:
                    continue
                rect = fitz.Rect(table.bbox)
                table_rects.append(rect)
                table_blocks.append({
                    "bbox": list(table.bbox),
                    "html": _rows_to_html(rows),
                    "rows": rows,
                })
        except Exception:
            pass

    # Drop text fragments that belong to a detected table.
    kept = []
    for b in raw_blocks:
        r = fitz.Rect(b["bbox"])
        if any((r & tr).get_area() > r.get_area() * 0.5 for tr in table_rects if not tr.is_empty):
            continue
        kept.append(b)

    median = sorted(sizes)[len(sizes) // 2] if sizes else 10.0
    ordered = _reading_order(kept + table_blocks, w, h)

    blocks: list[Block] = []
    for i, b in enumerate(ordered):
        bbox = _norm(b["bbox"], w, h)
        if "html" in b:
            blocks.append(Block(id=f"p{page_no}b{i}", page=page_no, type="table",
                                bbox=bbox, reading_order=i, content=b["html"],
                                html=b["html"], source="native_text", confidence=0.98))
        else:
            text = b["text"]
            btype = "text"
            if b["size"] > median * 1.25 and len(text) < 120:
                btype = "title"
            blocks.append(Block(id=f"p{page_no}b{i}", page=page_no, type=btype,
                                bbox=bbox, reading_order=i, content=text,
                                source="native_text", confidence=0.99))

    # Image regions (as figures), appended after text in reading order.
    try:
        for info in page.get_image_info():
            r = fitz.Rect(info["bbox"]) & page.rect
            if r.is_empty or r.get_area() < page.rect.get_area() * 0.02:
                continue
            i = len(blocks)
            blocks.append(Block(id=f"p{page_no}b{i}", page=page_no, type="figure",
                                bbox=_norm(list(r), w, h), reading_order=i,
                                content="", source="native_text", confidence=0.9,
                                meta={"image": True}))
    except Exception:
        pass

    return blocks


def render_page(page: fitz.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def render_pdf_page(path: Path, page_index: int, dpi: int) -> Image.Image:
    with fitz.open(path) as doc:
        return render_page(doc[page_index], dpi)
