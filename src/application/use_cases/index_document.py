from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict

from src.application.ports.vector_store import VectorStore
from src.domain.models import DocumentResult, DocumentStatus

logger = logging.getLogger(__name__)


def stable_document_id(final_path: Path | None, fallback_seed: str) -> str:
    """
    ID estable basado en el contenido de la imagen final.
    Reprocesar la misma imagen produce el mismo ID (upsert, no duplicado).
    """
    try:
        if final_path is not None and Path(final_path).exists():
            return hashlib.sha256(Path(final_path).read_bytes()).hexdigest()
    except OSError:
        pass
    return hashlib.sha256(fallback_seed.encode("utf-8")).hexdigest()


class IndexDocumentUseCase:
    """
    Caso de uso para indexar un documento procesado en la base vectorial.

    Este caso de uso recibe un DocumentResult y decide que texto
    se debe vectorizar, junto con su metadata.
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
            # ID estable basado en contenido (reprocesos no duplican)
            document_id = stable_document_id(
                result.final_path, fallback_seed=result.source_path.name
            )

            text = self._build_index_text(result)
            metadata = self._build_index_metadata(result)

            self._vector_store.upsert_document(
                document_id=document_id,
                text=text,
                metadata=metadata,
            )

            logger.info(
                "Documento indexado correctamente. ID: %s. Archivo: %s",
                document_id,
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
        
        CAMBIO: Solo indexa la estructura limpia (no raw_text).
        El raw_text se conserva en el sidecar JSON y se lee en tiempo de consulta RAG.
        """

        payload = result.payload if isinstance(result.payload, dict) else {}

        structure = payload.get("estructura", {})
        if not isinstance(structure, dict):
            structure = {}

        document_name = payload.get("documento", result.source_path.name)

        # Resumen estructurado limpio (discriminante, sin ruido)
        lines = [
            f"FACTURA {document_name}",
            f"proveedor: {structure.get('proveedor') or 'desconocido'}",
            f"nit: {structure.get('nit') or 'desconocido'}",
            f"fecha: {structure.get('fecha') or 'desconocida'}",
            f"numero_factura: {structure.get('numero_factura') or 'desconocido'}",
            f"moneda: {structure.get('moneda') or 'desconocida'}",
            f"subtotal: {structure.get('subtotal') or 'desconocido'}",
            f"impuestos: {structure.get('impuestos') or 'desconocido'}",
            f"total: {structure.get('total') or 'desconocido'}",
        ]

        # Items (si existen)
        items = structure.get("items") or []
        if items:
            item_desc = "; ".join(
                f"{str(i.get('descripcion') or '')[:80]} ({i.get('cantidad')} x {i.get('precio_unitario')})"
                for i in items[:5]
                if isinstance(i, dict)
            )
            lines.append(f"items: {item_desc}")

        # CUFE y enlace DIAN (si existen)
        if payload.get("cufe"):
            lines.append(f"CUFE: {payload['cufe']}")
        if payload.get("dian_url"):
            lines.append(f"dian_url: {payload['dian_url']}")

        # Observaciones (si existen)
        observaciones = structure.get("observaciones") or []
        if observaciones:
            lines.append(f"observaciones: {'; '.join(str(o) for o in observaciones[:3])}")

        # Validaciones (si existen)
        validaciones = payload.get("validaciones") or []
        if validaciones:
            lines.append(f"validaciones: {'; '.join(str(v) for v in validaciones)}")

        return "\n".join(lines)

    def _build_index_metadata(self, result: DocumentResult) -> Dict[str, Any]:
        """
        Construye metadata simple para búsquedas filtradas en FAISS.
        """

        payload = result.payload if isinstance(result.payload, dict) else {}

        structure = payload.get("estructura", {})
        if not isinstance(structure, dict):
            structure = {}

        final_file = ""
        sidecar_file = ""
        if result.final_path is not None:
            final_file = result.final_path.name
            sidecar_file = str(Path(final_file).with_suffix(".json").name)

        return {
            "document_id": result.document_id,
            "source_file": result.source_path.name,
            "final_file": final_file,
            "sidecar_file": sidecar_file,
            "status": result.status.value,
            "extractor": str(payload.get("extractor", "desconocido")),
            "llm_used": str(payload.get("llm_used", "desconocido")),
            "proveedor": str(structure.get("proveedor") or "desconocido"),
            "nit": str(structure.get("nit") or "desconocido"),
            "fecha": str(structure.get("fecha") or "desconocida"),
            "numero_factura": str(structure.get("numero_factura") or "desconocido"),
            "moneda": str(structure.get("moneda") or "desconocida"),
            "created_at": result.created_at.isoformat(),
        }