from __future__ import annotations

import json
import logging
from typing import Any, Dict

from src.application.ports.vector_store import VectorStore
from src.domain.models import DocumentResult, DocumentStatus

logger = logging.getLogger(__name__)


class IndexDocumentUseCase:
    """
    Caso de uso para indexar un documento procesado en la base vectorial.

    Este caso de uso recibe un DocumentResult y decide que texto
    se debe vectorizar, junto con su metadata.

    PENDIENTE OCR / IA:
    - Cuando exista OCR real, el payload puede incluir:
      - raw_text
      - qr
      - preprocessed_image_path
      - ocr_confidence
    - Cuando exista LLM real, el payload puede incluir:
      - structured_invoice
      - llm_provider
      - llm_confidence
    - Este caso de uso debera adaptar esos campos al texto vectorizable.
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store

    def execute(self, result: DocumentResult) -> bool:
        """
        Indexa un documento si el procesamiento fue exitoso.

        Devuelve:
        - True si se indexo correctamente.
        - False si no se indexo.
        """

        if result.status != DocumentStatus.SUCCESS:
            logger.info(
                "No se indexa documento porque no fue exitoso. Estado: %s. Archivo: %s",
                result.status.value,
                result.source_path,
            )
            return False

        try:
            text = self._build_index_text(result)
            metadata = self._build_index_metadata(result)

            self._vector_store.upsert_document(
                document_id=result.document_id,
                text=text,
                metadata=metadata,
            )

            logger.info(
                "Documento indexado correctamente. ID: %s. Archivo: %s",
                result.document_id,
                result.source_path,
            )

            return True

        except Exception:
            logger.exception(
                "Error indexando documento. ID: %s. Archivo: %s",
                result.document_id,
                result.source_path,
            )
            return False

    def _build_index_text(self, result: DocumentResult) -> str:
        """
        Construye el texto que se va a vectorizar.

        Para el stub actual se usa el payload basico.
        Cuando exista OCR real, aqui se combinara:
        - JSON estructurado
        - texto OCR crudo
        - datos QR
        """

        payload = result.payload if isinstance(result.payload, dict) else {}

        structure = payload.get("estructura", {})
        if not isinstance(structure, dict):
            structure = {}

        document_name = payload.get("documento", result.source_path.name)

        lines = [
            f"Documento: {document_name}",
            f"Estado: {result.status.value}",
            f"Extractor: {payload.get('extractor', 'desconocido')}",
            "Extracto estructurado:",
            f"proveedor: {structure.get('proveedor') or 'desconocido'}",
            f"nit: {structure.get('nit') or 'desconocido'}",
            f"fecha: {structure.get('fecha') or 'desconocida'}",
            f"moneda: {structure.get('moneda') or 'desconocida'}",
            f"subtotal: {structure.get('subtotal') or 'desconocido'}",
            f"impuestos: {structure.get('impuestos') or 'desconocido'}",
            f"total: {structure.get('total') or 'desconocido'}",
        ]

        # PENDIENTE OCR:
        # Cuando exista EasyOCR, el payload deberia incluir raw_text.
        raw_text = payload.get("raw_text")
        if raw_text:
            lines.append(f"Texto OCR: {str(raw_text)[:2000]}")

        # PENDIENTE QR:
        # Cuando exista pyzbar, el payload deberia incluir qr.
        qr = payload.get("qr")
        if qr:
            try:
                qr_text = json.dumps(qr, ensure_ascii=False)
                lines.append(f"Datos QR: {qr_text[:1000]}")
            except Exception:
                lines.append("Datos QR: no serializables")

        return "\n".join(lines)

    def _build_index_metadata(self, result: DocumentResult) -> Dict[str, Any]:
        """
        Construye metadata simple para busquedas filtradas.

        ChromaDB trabaja mejor con valores simples:
        - str
        - int
        - float
        - bool
        """

        payload = result.payload if isinstance(result.payload, dict) else {}

        structure = payload.get("estructura", {})
        if not isinstance(structure, dict):
            structure = {}

        final_file = ""
        if result.final_path is not None:
            final_file = result.final_path.name

        return {
            "document_id": result.document_id,
            "source_file": result.source_path.name,
            "final_file": final_file,
            "status": result.status.value,
            "extractor": str(payload.get("extractor", "desconocido")),
            "proveedor": str(structure.get("proveedor") or "desconocido"),
            "nit": str(structure.get("nit") or "desconocido"),
            "fecha": str(structure.get("fecha") or "desconocida"),
            "moneda": str(structure.get("moneda") or "desconocida"),
            "total": str(structure.get("total") or "desconocido"),
            "created_at": result.created_at.isoformat(),
        }