"""Native parsers for office/text formats.

These never call the VLM: the structure is already machine-readable, and OCR
would only add errors. Dependencies are imported lazily so a missing optional
package does not break the rest of the pipeline.
"""

from __future__ import annotations

import html as _html
from pathlib import Path

from .ir import Block, Document, Page
from .pdfio import sha256_file


def _table_html(rows: list[list[str]]) -> str:
    body = []
    for row in rows:
        body.append("<tr>" + "".join(
            f"<td>{_html.escape(c or '').strip()}</td>" for c in row) + "</tr>")
    return "<table>\n" + "\n".join(body) + "\n</table>"


def _emit(blocks: list[Block], page_no: int, btype: str, content: str,
          *, html: str | None = None, meta: dict | None = None) -> None:
    if not (content or html):
        return
    blocks.append(Block(id=f"p{page_no}b{len(blocks)}", page=page_no, type=btype,
                        bbox=[0, 0, 1000, 1000], reading_order=len(blocks),
                        content=content, html=html, source="native_office",
                        confidence=0.99, meta=meta or {}))


def parse_docx(path: Path) -> Document:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(str(path))
    doc = Document(path=str(path), sha256=sha256_file(path), doc_type="docx")
    page = Page(number=1, kind="native")
    blocks: list[Block] = []

    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, document)
            text = para.text.strip()
            if not text:
                continue
            style = (para.style.name or "") if para.style is not None else ""
            btype = "title" if style.lower().startswith("heading") else "text"
            _emit(blocks, 1, btype, text, meta={"style": style})
        elif tag == "tbl":
            table = Table(child, document)
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            _emit(blocks, 1, "table", "", html=_table_html(rows))

    page.blocks = blocks
    page.mode = "native"
    doc.pages.append(page)
    return doc


def parse_pptx(path: Path) -> Document:
    from pptx import Presentation

    prs = Presentation(str(path))
    doc = Document(path=str(path), sha256=sha256_file(path), doc_type="pptx")
    for i, slide in enumerate(prs.slides, 1):
        page = Page(number=i, kind="native")
        blocks: list[Block] = []
        for shape in slide.shapes:
            if shape.has_text_frame and shape.text_frame.text.strip():
                _emit(blocks, i, "text", shape.text_frame.text.strip(),
                      meta={"shape": shape.name})
            if getattr(shape, "has_table", False) and shape.has_table:
                rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
                _emit(blocks, i, "table", "", html=_table_html(rows))
        page.blocks = blocks
        page.mode = "native"
        doc.pages.append(page)
    return doc


def parse_xlsx(path: Path) -> Document:
    import openpyxl

    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    doc = Document(path=str(path), sha256=sha256_file(path), doc_type="xlsx")
    try:
        for i, sheet in enumerate(wb.worksheets, 1):
            rows = [["" if c is None else str(c) for c in row]
                    for row in sheet.iter_rows(values_only=True)]
            rows = [r for r in rows if any(c.strip() for c in r)]
            page = Page(number=i, kind="native", mode="native")
            if rows:
                page.blocks.append(Block(
                    id=f"p{i}b0", page=i, type="table", bbox=[0, 0, 1000, 1000],
                    reading_order=0, content="", html=_table_html(rows),
                    source="native_office", confidence=0.99,
                    meta={"sheet": sheet.title, "rows": len(rows)}))
            page.warnings.append({"code": "sheet", "severity": "info",
                                  "message": f"Sheet '{sheet.title}'"})
            doc.pages.append(page)
    finally:
        wb.close()
    return doc


def parse_text(path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace")
    doc = Document(path=str(path), sha256=sha256_file(path), doc_type="text")
    page = Page(number=1, kind="native", mode="native")
    page.blocks.append(Block(id="p1b0", page=1, type="text", bbox=[0, 0, 1000, 1000],
                             reading_order=0, content=text, source="native_text",
                             confidence=1.0))
    doc.pages.append(page)
    return doc
