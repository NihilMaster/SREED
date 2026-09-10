from __future__ import annotations

import time
from pathlib import Path

from src.domain.models import DocumentResult, DocumentStatus


class StubDocumentProcessor:
    """
    Procesador temporal para validar el flujo.

    Este stub sera reemplazado por el adaptador real:
    OpenCV -> EasyOCR -> pyzbar -> Qwen.
    """

    def process(self, source_path: Path) -> DocumentResult:
        if not source_path.exists():
            raise FileNotFoundError(f"No se encontro el archivo: {source_path}")

        # Simulacion breve de procesamiento.
        time.sleep(0.3)

        size = source_path.stat().st_size

        payload = {
            "documento": source_path.name,
            "tamano_bytes": size,
            "extractor": "stub",
            "estructura": {
                "proveedor": None,
                "nit": None,
                "fecha": None,
                "moneda": None,
                "subtotal": None,
                "impuestos": None,
                "total": None,
            },
            "nota": "Este stub sera reemplazado por OpenCV + EasyOCR + Qwen.",
        }

        return DocumentResult(
            source_path=source_path,
            status=DocumentStatus.SUCCESS,
            message="Procesamiento simulado completado.",
            payload=payload,
        )