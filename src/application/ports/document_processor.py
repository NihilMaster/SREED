from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.domain.models import DocumentResult


class DocumentProcessor(Protocol):
    """
    Puerto para cualquier motor de procesamiento de documentos.

    Implementaciones futuras:
    - OpenCV + EasyOCR + pyzbar + Qwen.
    - Gemini opcional.
    - Stub actual.
    """

    def process(self, source_path: Path) -> DocumentResult:
        ...