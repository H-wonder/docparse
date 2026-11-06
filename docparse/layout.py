"""PP-DocLayoutV3 wrapper (optional).

Runs on CPU by default so the GPU stays dedicated to the 0.8B VLM. Returns
regions in the detector's reading order with normalized 0-1000 boxes.
"""

from __future__ import annotations

import threading
from typing import Any

from PIL import Image

from .config import LABEL_TASK

_LOCK = threading.Lock()
_MODEL = None


def available() -> bool:
    try:
        import paddlex  # noqa: F401
        return True
    except Exception:
        return False


def _bucket(label: str) -> str | None:
    for task, labels in LABEL_TASK.items():
        if label in labels:
            return task
    return None


def _score_label(label: str, bucket: str) -> str:
    if label in {"doc_title", "paragraph_title", "title", "section_header"}:
        return "title"
    return {"text": "text", "table": "table", "formula": "formula", "skip": "figure"}.get(bucket, "text")


def detect(image: Image.Image, *, device: str = "cpu", threshold: float = 0.3) -> list[dict[str, Any]]:
    """Return [{label, bucket, bbox, score}] in reading order."""
    global _MODEL
    import numpy as np
    import paddlex as pdx

    with _LOCK:
        if _MODEL is None:
            _MODEL = pdx.create_model("PP-DocLayoutV3", threshold=threshold, device=device)
        result = list(_MODEL.predict(np.asarray(image)))[0]

    width, height = image.size
    boxes = []
    if hasattr(result, "get") and result.get("boxes") is not None:
        boxes = result["boxes"]
    else:
        inner = result.get("res") if hasattr(result, "get") else None
        boxes = inner.get("boxes", []) if inner else []

    regions: list[dict[str, Any]] = []
    for box in boxes:
        label = str(box.get("label", ""))
        coord = box.get("coordinate", [])
        if len(coord) != 4:
            continue
        bucket = _bucket(label)
        if bucket is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in coord]
        x1 = max(0.0, min(x1, width)); x2 = max(0.0, min(x2, width))
        y1 = max(0.0, min(y1, height)); y2 = max(0.0, min(y2, height))
        if x2 <= x1 or y2 <= y1:
            continue
        regions.append({
            "label": label,
            "bucket": _score_label(label, bucket),
            "bbox": [int(x1 / width * 1000), int(y1 / height * 1000),
                     int(x2 / width * 1000), int(y2 / height * 1000)],
            "score": float(box.get("score", 0.0)),
        })
    return regions
