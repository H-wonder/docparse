"""Image normalization aligned with the model's documented pixel bounds."""

from __future__ import annotations

import io

from PIL import Image, ImageOps

from .config import MAX_EDGE, MIN_PIXELS


def open_image(raw: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(raw))
    image = ImageOps.exif_transpose(image)          # phone photos
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def fit(image: Image.Image, max_edge: int = MAX_EDGE) -> Image.Image:
    """Downscale to the supported long-edge bound, preserving aspect ratio."""
    if max(image.size) > max_edge:
        image = image.copy()
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return image


def ensure_min_pixels(image: Image.Image, min_pixels: int = MIN_PIXELS) -> Image.Image:
    """Upscale tiny crops: the vision tower has a lower pixel bound too."""
    w, h = image.size
    if w * h >= min_pixels:
        return image
    scale = (min_pixels / max(1, w * h)) ** 0.5
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                        Image.Resampling.BICUBIC)


def prepare(raw: bytes, max_edge: int = MAX_EDGE) -> bytes:
    """Decode → EXIF-rotate → RGB → fit bounds. Returns PNG bytes."""
    image = fit(ensure_min_pixels(open_image(raw)), max_edge)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def prepare_image(image: Image.Image, max_edge: int = MAX_EDGE) -> bytes:
    image = fit(ensure_min_pixels(image.convert("RGB")), max_edge)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()
