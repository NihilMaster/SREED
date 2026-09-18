from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.application.ports.document_processor import DocumentProcessor
from src.application.ports.file_system import FileSystemService
from src.application.ports.llm_provider import LLMProvider
from src.domain.dian_qr import DIAN_QR_URL, apply_qr_fiscal_crosscheck, parse_dian_qr
from src.domain.models import DocumentResult, DocumentStatus
from src.domain.validation import normalize_invoice_structure, validate_invoice_structure

logger = logging.getLogger(__name__)


class ProcessDocumentUseCase:
    """
    Caso de uso: procesar un documento entrante.

    Flujo:
    1. Validar existencia, extension y estabilidad.
    2. OCR (puerto DocumentProcessor).
    3. Estructuracion (puerto LLMProvider).
    4. Cross-check con QR fiscal DIAN si existe (fuente autoritativa).
    5. Normalizacion y validacion deterministica.
    6. Mover a procesados o fallidos y escribir JSON lateral.
    """

    def __init__(
        self,
        processor: DocumentProcessor,
        llm_provider: LLMProvider,
        file_system: FileSystemService,
    ) -> None:
        self._processor = processor
        self._llm_provider = llm_provider
        self._file_system = file_system

    def execute(self, source_path: Path) -> DocumentResult:
        logger.info("Iniciando procesamiento de: %s", source_path)

        try:
            if not source_path.exists():
                return DocumentResult(
                    source_path=source_path,
                    status=DocumentStatus.FAILED,
                    message="El archivo ya no existe.",
                )

            if self._file_system.is_ignorable(source_path):
                return DocumentResult(
                    source_path=source_path,
                    status=DocumentStatus.PENDING,
                    message="Archivo ignorado por regla de entrada.",
                )

            if not self._file_system.is_supported(source_path):
                result = DocumentResult(
                    source_path=source_path,
                    status=DocumentStatus.UNSUPPORTED,
                    message="Extensión no soportada.",
                )
                failed_path = self._file_system.move_to_failed(source_path)
                result.final_path = failed_path
                self._file_system.write_result(failed_path, result)
                return result

            if not self._file_system.wait_until_stable(source_path):
                result = DocumentResult(
                    source_path=source_path,
                    status=DocumentStatus.UNSTABLE,
                    message="El archivo no alcanzó estabilidad dentro del tiempo esperado.",
                )
                failed_path = self._file_system.move_to_failed(source_path)
                result.final_path = failed_path
                self._file_system.write_result(failed_path, result)
                return result

            # 1. OCR
            logger.info("Ejecutando OCR...")
            result = self._processor.process(source_path)

            # 2. Estructuracion con LLM
            if result.status == DocumentStatus.SUCCESS:
                raw_text = result.payload.get("raw_text", "")
                if raw_text.strip():
                    logger.info("Estructurando con Qwen...")
                    result = self._llm_provider.process(result)
                else:
                    result.status = DocumentStatus.FAILED
                    result.message = "No se pudo extraer texto del documento"

            if result.status != DocumentStatus.SUCCESS:
                failed_path = self._file_system.move_to_failed(source_path)
                result.final_path = failed_path
                self._file_system.write_result(failed_path, result)
                return result

            # 3. Cross-check con QR fiscal DIAN (fuente autoritativa firmada)
            qr_fiscal = self._first_dian_qr(result.payload.get("qr", []))
            if qr_fiscal:
                estructura, cambios = apply_qr_fiscal_crosscheck(
                    result.payload.get("estructura", {}), qr_fiscal
                )
                result.payload["estructura"] = estructura
                result.payload["qr_fiscal"] = qr_fiscal
                if cambios:
                    result.payload["qr_crosscheck"] = cambios

                cufe = qr_fiscal.get("CUFE")
                if cufe:
                    result.payload["cufe"] = cufe
                    result.payload["cufe_fuente"] = "qr_fiscal"
                    result.payload["dian_url"] = DIAN_QR_URL + cufe

                logger.info("Cross-check fiscal aplicado desde QR. Campos: %s", cambios)

            # 4. Normalizacion y validacion deterministica
            estructura = result.payload.get("estructura")
            if isinstance(estructura, dict):
                estructura = normalize_invoice_structure(estructura)
                result.payload["estructura"] = estructura

                issues = validate_invoice_structure(estructura)
                result.payload["validaciones"] = issues
                if issues:
                    logger.warning("Observaciones de validacion para %s: %s", source_path.name, issues)

            # 5. Movimiento y persistencia
            final_path = self._file_system.move_to_processed(source_path)
            result.final_path = final_path
            self._file_system.write_result(final_path, result)

            logger.info(
                "Documento procesado. Estado=%s. Origen=%s. Destino=%s",
                result.status.value, source_path, final_path,
            )
            return result

        except Exception as exc:
            logger.exception("Error no controlado procesando documento: %s", source_path)
            try:
                failed_path = self._file_system.move_to_failed(source_path)
                error_result = DocumentResult(
                    source_path=source_path,
                    final_path=failed_path,
                    status=DocumentStatus.FAILED,
                    message=str(exc),
                    payload={"exception": exc.__class__.__name__},
                )
                self._file_system.write_result(failed_path, error_result)
                return error_result
            except Exception:
                logger.exception("No fue posible mover el archivo a fallidas.")
                return DocumentResult(
                    source_path=source_path,
                    status=DocumentStatus.FAILED,
                    message=str(exc),
                )

    @staticmethod
    def _first_dian_qr(qr_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for qr in qr_list or []:
            parsed = parse_dian_qr(qr.get("data", ""))
            if parsed:
                return parsed
        return None