"""Canonical document representation.

Every renderer (Markdown / JSON / future HTML) is a serializer over this IR.
Nothing downstream should depend on raw model output directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Block:
    id: str
    page: int                              # 1-based
    type: str                              # title|text|table|formula|figure|caption|seal|chart|skip
    bbox: list[float] = field(default_factory=list)  # normalized 0-1000 [x1,y1,x2,y2]
    reading_order: int = 0
    content: str = ""                      # markdown / OTSL / LaTeX / plain text
    html: str | None = None                # tables only
    source: str = "vlm"                    # native_text|vlm_page|vlm_region|fallback
    confidence: float = 1.0
    warnings: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Page:
    number: int
    width: int = 0
    height: int = 0
    kind: str = "unknown"                  # native|scanned|hybrid|image
    blocks: list[Block] = field(default_factory=list)
    mode: str = ""                         # page|region|native
    warnings: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: int = 0
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "width": self.width,
            "height": self.height,
            "kind": self.kind,
            "mode": self.mode,
            "warnings": self.warnings,
            "latency_ms": self.latency_ms,
            "usage": self.usage,
            "blocks": [b.to_dict() for b in self.blocks],
        }


@dataclass
class Document:
    path: str
    sha256: str = ""
    doc_type: str = "unknown"
    pages: list[Page] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_review(self) -> bool:
        return any(w.get("severity") == "high" for w in self.warnings) or any(
            w.get("severity") == "high" for p in self.pages for w in p.warnings
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "doc_type": self.doc_type,
            "needs_review": self.needs_review,
            "warnings": self.warnings,
            "meta": self.meta,
            "pages": [p.to_dict() for p in self.pages],
        }
