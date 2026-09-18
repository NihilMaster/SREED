from __future__ import annotations

import logging
from pathlib import Path

from src.application.use_cases.index_document import IndexDocumentUseCase
from src.application.use_cases.process_document import ProcessDocumentUseCase
from src.domain.models import DocumentResult, DocumentStatus

logger = logging.getLogger(__name__)


class ProcessAndIndexDocumentUseCase:
    """
    Caso de uso compuesto.

    Flujo:
    1. Procesa un documento usando ProcessDocumentUseCase (OCR + Qwen).
    2. Si el procesamiento fue exitoso, lo indexa usando IndexDocumentUseCase.

    Este caso de uso existe para no modificar ProcessDocumentUseCase.
    También mantiene separadas las responsabilidades:
    - procesar documento (OCR + LLM)
    - indexar documento (FAISS)
    """

    def __init__(
        self,
        process_document: ProcessDocumentUseCase,
        index_document: IndexDocumentUseCase,
    ) -> None:
        self._process_document = process_document
        self._index_document = index_document

    def execute(self, source_path: Path) -> DocumentResult:
        result = self._process_document.execute(source_path)

        if result.status == DocumentStatus.SUCCESS:
            indexed = self._index_document.execute(result)

            if indexed:
                logger.info(
                    "Documento procesado e indexado. ID: %s. Archivo: %s",
                    result.document_id,
                    result.source_path,
                )
            else:
                logger.warning(
                    "El documento se procesó correctamente, pero falló la indexación. "
                    "ID: %s. Archivo: %s",
                    result.document_id,
                    result.source_path,
                )
        else:
            logger.info(
                "Documento no indexado porque el procesamiento no fue exitoso. "
                "Estado: %s. Archivo: %s",
                result.status.value if hasattr(result.status, 'value') else result.status,
                result.source_path,
            )

        return result