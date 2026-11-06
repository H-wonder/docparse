"""End-to-end parsing pipeline with automatic escalation.

Ladder (all automatic, no human step):
  native text layer → whole-page VLM → repetition retry → region VLM → flag
Native PDF text is never sent through OCR if its text layer is healthy.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import fitz

from postprocess.markdown_merge import merge_page_to_markdown

from . import backend, imageops, layout, office, pdfio, quality
from .config import (
    PROMPTS,
    REGION_MAX_TOKENS,
    RETRY_PRESENCE_PENALTY,
    ParseOptions,
)
from .ir import Block, Document, Page

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_EXT = {".pdf"}


class Parser:
    def __init__(self, opts: ParseOptions | None = None):
        self.opts = opts or ParseOptions()
        self._layout_ok = layout.available() if self.opts.mode != "page" else False

    # ── public ──────────────────────────────────────────────────────────────
    def parse(self, path: str | Path) -> Document:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in PDF_EXT:
            return self._parse_pdf(path)
        if suffix in IMAGE_EXT:
            return self._parse_image_file(path)
        native = {
            ".docx": office.parse_docx,
            ".pptx": office.parse_pptx,
            ".xlsx": office.parse_xlsx,
            ".txt": office.parse_text,
            ".md": office.parse_text,
        }
        if suffix in native:
            return native[suffix](path)
        raise ValueError(
            f"Unsupported input: {suffix}. Supported: PDF, images, "
            f"{', '.join(sorted(native))}")

    # ── images ──────────────────────────────────────────────────────────────
    def _parse_image_file(self, path: Path) -> Document:
        raw = path.read_bytes()
        image = imageops.open_image(raw)
        doc = Document(path=str(path), sha256=pdfio.sha256_file(path), doc_type="image")
        page = Page(number=1, width=image.width, height=image.height, kind="image")
        try:
            blocks, mode, warns, latency, usage = self._ocr(image, 1)
        except Exception as exc:
            blocks, mode, warns, latency, usage = [], "failed", [
                quality.warning("page_failed", "high", str(exc))], 0, {}
        page.blocks, page.mode = blocks, mode
        page.warnings.extend(warns)
        page.latency_ms, page.usage = latency, usage
        if self.opts.kie_fields:
            try:
                kie, kwarns = self._kie(image, 1, _blocks_text(page.blocks))
            except Exception as exc:
                kie = {"json": None, "raw": "", "grounding": {}}
                kwarns = [quality.warning("kie_failed", "medium", str(exc))]
            doc.meta["kie"] = [{"page": 1, **kie}]
            page.warnings.extend(kwarns)
        doc.pages.append(page)
        doc.warnings.extend({**w, "page": 1} for w in page.warnings
                            if w.get("severity") in {"high", "medium"})
        return doc

    # ── pdf ─────────────────────────────────────────────────────────────────
    def _parse_pdf(self, path: Path) -> Document:
        doc = Document(path=str(path), sha256=pdfio.sha256_file(path), doc_type="pdf")
        with fitz.open(path) as pdf:
            total = len(pdf)
            limit = total if self.opts.max_pages is None else min(total, self.opts.max_pages)
            doc.meta["pages_total"] = total
            if limit < total:
                doc.meta["pages_parsed"] = limit

            for index in range(limit):
                fitz_page = pdf[index]
                probe = pdfio.probe_page(fitz_page, index + 1)
                page = Page(number=index + 1, width=int(probe.width),
                            height=int(probe.height), kind=probe.kind)
                page_image = None

                if probe.kind in {"native", "hybrid"} and probe.chars >= 10 and probe.bad_ratio <= 0.05:
                    page.blocks = pdfio.extract_native(fitz_page, index + 1)
                    page.mode = "native"
                    page.warnings.append(quality.warning(
                        "native_text", "info",
                        f"Used embedded text layer ({probe.chars} chars)."))
                elif probe.kind == "blank":
                    page.mode = "blank"
                    page.warnings.append(quality.warning(
                        "blank_page", "info", "No text and no image content."))
                else:
                    page_image = pdfio.render_page(fitz_page, self.opts.dpi)
                    try:
                        blocks, mode, warns, latency, usage = self._ocr(page_image, index + 1)
                    except Exception as exc:  # one bad page must not kill the document
                        blocks = []
                        mode = "failed"
                        warns = [quality.warning("page_failed", "high", str(exc))]
                        latency, usage = 0, {}
                    page.blocks, page.mode = blocks, mode
                    page.warnings.extend(warns)
                    page.latency_ms, page.usage = latency, usage
                    page.warnings.append(quality.warning(
                        "ocr_path", "info",
                        f"Text layer unusable ({probe.chars} chars, "
                        f"{probe.bad_ratio:.1%} bad); rendered at {self.opts.dpi} DPI."))

                if self.opts.kie_fields and probe.kind != "blank":
                    try:
                        if page_image is None:
                            page_image = pdfio.render_page(fitz_page, self.opts.dpi)
                        kie, kwarns = self._kie(page_image, page.number, _blocks_text(page.blocks))
                    except Exception as exc:
                        kie = {"json": None, "raw": "", "grounding": {}}
                        kwarns = [quality.warning("kie_failed", "medium", str(exc))]
                    doc.meta.setdefault("kie", []).append({"page": page.number, **kie})
                    page.warnings.extend(kwarns)

                doc.pages.append(page)
                for w in page.warnings:
                    if w.get("severity") in {"high", "medium"}:
                        doc.warnings.append({**w, "page": page.number})
        return doc

    # ── key information extraction ──────────────────────────────────────────
    def _kie(self, image, page_no: int, page_text: str):
        from .config import kie_prompt
        result = backend.chat(imageops.prepare_image(image), kie_prompt(self.opts.kie_fields),
                              min(self.opts.max_tokens, 2048))
        data, error = quality.parse_kie(result["text"], self.opts.kie_fields)
        warns: list[dict] = []
        if data is None:
            warns.append(quality.warning("kie_invalid", "high",
                                         f"KIE output unusable ({error}).", page=page_no))
            return {"json": None, "raw": result["text"], "grounding": {}}, warns

        grounding = quality.ground_fields(data, page_text)
        for field, outcome in grounding.items():
            if outcome["reason"] == "not_found_in_ocr_text":
                value = str(data.get(field, ""))
                high = sum(ch.isdigit() for ch in value) >= 5
                warns.append(quality.warning(
                    "field_not_grounded", "high" if high else "medium",
                    f"KIE field '{field}' does not appear in the OCR text "
                    f"(possible hallucination or normalization).",
                    field=field, value=value))
            elif outcome["reason"] == "empty":
                warns.append(quality.warning("field_missing", "info",
                                             f"KIE field '{field}' is empty.", field=field))
        return {"json": data, "raw": result["text"], "grounding": grounding}, warns

    # ── OCR core ────────────────────────────────────────────────────────────
    def _ocr(self, image, page_no: int) -> tuple[list[Block], str, list[dict], int, dict]:
        t0 = time.perf_counter()
        mode = self.opts.mode
        if mode == "region" and not self._layout_ok:
            mode = "page"
            warns_pre = [quality.warning(
                "layout_unavailable", "medium",
                "Region mode requested but PaddleX/PP-DocLayoutV3 is not installed; used page mode.")]
        else:
            warns_pre = []

        if mode == "region":
            try:
                blocks, warns, usage = self._ocr_regions(image, page_no)
                latency = int((time.perf_counter() - t0) * 1000)
                return blocks, "region", warns_pre + warns, latency, usage
            except Exception as exc:  # layout runtime/model failure → page fallback
                warns_pre.append(quality.warning(
                    "region_failed", "medium",
                    f"Region parsing failed ({exc}); fell back to whole-page mode."))

        # whole page
        blocks, warns, usage = self._ocr_page(image, page_no, mode="page")
        latency = int((time.perf_counter() - t0) * 1000)

        if (self.opts.mode == "auto" and self._layout_ok
                and any(w.get("severity") == "high" for w in warns)):
            # Page mode failed a hard gate → escalate to region parsing once.
            try:
                rblocks, rwarns, rusage = self._ocr_regions(image, page_no)
                rlatency = int((time.perf_counter() - t0) * 1000)
                if rblocks and not any(w.get("severity") == "high" for w in rwarns):
                    rwarns = warns_pre + [quality.warning(
                        "escalated_to_region", "info",
                        "Whole-page output failed quality gates; region parsing adopted.")] + rwarns
                    return rblocks, "region", rwarns, rlatency, rusage
                warns = warns + rwarns
            except Exception as exc:  # layout/runtime failure must not kill the page
                warns.append(quality.warning("region_escalation_failed", "medium", str(exc)))

        return blocks, "page", warns_pre + warns, latency, usage

    def _ocr_page(self, image, page_no: int, *, mode: str = "page"):
        png = imageops.prepare_image(image)
        try:
            result = backend.chat(png, PROMPTS["document"], self.opts.max_tokens)
        except backend.ContextOverflow:
            # Dense page + vision tokens can exceed the context window; shrink the
            # output budget before giving up (region mode is the next rung).
            result = backend.chat(png, PROMPTS["document"],
                                  max(1024, self.opts.max_tokens // 2))
        raw = result["text"]
        # Gate on the raw response: finalize() truncates runaway repeats, which
        # would otherwise hide the very signal the gate is looking for.
        qa = quality.assess(raw, result["finish_reason"], mode=mode)
        text = quality.finalize(raw)

        if qa["needs_retry"] and self.opts.retry_repetition:
            try:
                retry = backend.chat(png, PROMPTS["document"], self.opts.max_tokens,
                                     presence_penalty=RETRY_PRESENCE_PENALTY)
                rraw = retry["text"]
                rqa = quality.assess(rraw, retry["finish_reason"], mode=mode)
                if rqa["score"] >= qa["score"]:
                    text, qa = quality.finalize(rraw), rqa
                    qa["warnings"].append(quality.warning(
                        "repetition_retry", "info", "Retried with presence_penalty=1.5."))
                    result = {**retry, "text": text}
            except backend.BackendError:
                pass

        if qa["suspicious"]:
            # Still looping after the retry → a hard gate, so region mode gets a chance.
            qa["warnings"].append(quality.warning(
                "repetition_unresolved", "high",
                "Output is still repetitive after the automatic retry."))

        coverage = quality.coverage_warning(text, image)
        if coverage is not None:
            qa["warnings"].append(coverage)
            qa["score"] = max(0.0, qa["score"] - 0.4)

        blocks = [Block(id=f"p{page_no}b0", page=page_no, type="page",
                        bbox=[0, 0, 1000, 1000], reading_order=0,
                        content=text, source="vlm_page",
                        confidence=qa["score"], warnings=qa["warnings"],
                        meta={"finish_reason": result.get("finish_reason")})]
        return blocks, qa["warnings"], result.get("usage", {})

    def _ocr_regions(self, image, page_no: int):
        fitted = imageops.fit(image)
        regions = layout.detect(fitted, device=self.opts.layout_device)
        if not regions:
            raise RuntimeError("layout returned no usable regions")

        rows: list[dict[str, Any]] = []
        blocks: list[Block] = []
        all_warnings: list[dict] = []
        usage_total: dict[str, int] = {}

        for i, region in enumerate(regions):
            bbox = region["bbox"]
            kind = region["bucket"]
            if kind == "figure":
                blocks.append(Block(id=f"p{page_no}b{i}", page=page_no, type="figure",
                                    bbox=bbox, reading_order=i, content="",
                                    source="layout", confidence=region.get("score", 0)))
                rows.append({"index": i, "label": region["label"], "bbox_2d": bbox,
                             "merged": "", "score": region.get("score", 0)})
                continue

            crop = _crop_normalized(image, bbox)
            prompt = PROMPTS.get(kind, PROMPTS["text"])
            budget = min(self.opts.max_tokens, REGION_MAX_TOKENS.get(kind, 1024))
            result = backend.chat(crop, prompt, budget)
            text = result["text"]
            qa = quality.assess(text, result["finish_reason"], mode="region")

            if qa["needs_retry"] and self.opts.retry_repetition:
                try:
                    retry = backend.chat(crop, prompt, budget,
                                         presence_penalty=RETRY_PRESENCE_PENALTY)
                    rqa = quality.assess(retry["text"], retry["finish_reason"], mode="region")
                    if rqa["score"] >= qa["score"]:
                        text, qa = retry["text"], rqa
                        qa["warnings"].append(quality.warning(
                            "repetition_retry", "info", "Region retried with presence_penalty=1.5."))
                except backend.BackendError:
                    pass

            html = None
            if kind == "table":
                converted, ok = quality.convert_table(text)
                if ok:
                    html, text = converted, converted
                else:
                    qa["warnings"].append(quality.warning(
                        "table_parse_failed", "medium", "Kept raw OTSL table."))

            for k, v in (result.get("usage") or {}).items():
                if isinstance(v, int):
                    usage_total[k] = usage_total.get(k, 0) + v

            blocks.append(Block(
                id=f"p{page_no}b{i}", page=page_no,
                type={"text": "text", "title": "title", "table": "table",
                      "formula": "formula"}.get(kind, "text"),
                bbox=bbox, reading_order=i, content=text, html=html,
                source="vlm_region", confidence=qa["score"], warnings=qa["warnings"],
                meta={"layout_label": region["label"], "layout_score": region.get("score", 0)},
            ))
            all_warnings.extend(qa["warnings"])
            rows.append({"index": i, "label": region["label"], "bbox_2d": bbox,
                         "merged": text, "score": region.get("score", 0)})

        try:
            markdown = merge_page_to_markdown(rows)
        except Exception:
            markdown = "\n\n".join(b.content for b in blocks if b.content and b.type != "figure")

        # Stash the assembled markdown on a synthetic page-level block so writers
        # can use the upstream assembly (which handles titles, formulas, merging).
        blocks.insert(0, Block(id=f"p{page_no}md", page=page_no, type="page_markdown",
                               bbox=[0, 0, 1000, 1000], reading_order=-1,
                               content=markdown, source="vlm_region", confidence=1.0,
                               meta={"synthetic": True}))

        if not markdown.strip():
            all_warnings.append(quality.warning("empty_output", "high",
                                                "All regions returned empty text."))
        return blocks, all_warnings, usage_total


def _blocks_text(blocks: list[Block]) -> str:
    return "\n\n".join(b.content for b in blocks if b.content and b.type != "figure")


def _crop_normalized(image, bbox: list[float]):
    """Crop from the *full-resolution* page; layout boxes are normalized 0-1000."""
    from evaluation_pipeline.cropping import crop_region
    return crop_region(image, bbox)
