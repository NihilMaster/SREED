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
    1. Procesa un documento usando ProcessDocumentUseCase.
    2. Si el procesamiento fue exitoso, lo indexa usando IndexDocumentUseCase.

    Este caso de uso existe para no modificar ProcessDocumentUseCase.
    Tambien mantiene separadas las responsabilidades:
    - procesar documento
    - indexar documento
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
                    "El documento se proceso correctamente, pero fallo la indexacion. "
                    "ID: %s. Archivo: %s",
                    result.document_id,
                    result.source_path,
                )
        else:
            logger.info(
                "Documento no indexado porque el procesamiento no fue exitoso. "
                "Estado: %s. Archivo: %s",
                result.status.value,
                result.source_path,
            )

        return result