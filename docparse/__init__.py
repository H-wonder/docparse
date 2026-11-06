"""Local multimodal document parsing system built on Xiaomi-OCR-0.

The upstream repository is used as a library for postprocessing (OTSL→HTML,
repetition analysis, markdown assembly). It lives next to this package.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent / "Xiaomi-OCR-0"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

__version__ = "0.1.0"
