from typing import Dict, Any
import logging
import copy

from src.application.ports.llm_provider import LLMProvider
from src.domain.models import DocumentResult, DocumentStatus
from src.infrastructure.adapters.llm.qwen_ollama_provider import QwenOllamaProvider
from src.infrastructure.adapters.llm.gemini_vision_provider import GeminiVisionProvider

logger = logging.getLogger(__name__)

DIAN_QR_URL = "https://catalogo-vpfe.dian.gov.co/document/searchqr?documentkey="


class QwenGeminiStrategy(LLMProvider):
    """
    Estrategia hibrida:
    - Qwen (local): campos clave desde texto OCR (tarea corta).
    - Gemini Vision (nube): esquema completo (items, CUFE) y correccion de campos.
    Gemini nunca actua como proveedor unico.
    """

    OCR_CONFIDENCE_THRESHOLD = 0.6

    def __init__(self, qwen_provider: QwenOllamaProvider, gemini_provider: GeminiVisionProvider):
        self.qwen = qwen_provider
        self.gemini = gemini_provider
        logger.info("QwenGeminiStrategy inicializado (híbrido)")

    def process(self, document_result: DocumentResult) -> DocumentResult:
        ocr_confidence = document_result.payload.get("ocr_confidence", 0.0)
        logger.info(f"Estrategia híbrida: OCR confidence = {ocr_confidence:.2f}, umbral = {self.OCR_CONFIDENCE_THRESHOLD}")

        logger.info("Paso 1: Qwen extrayendo campos clave del texto OCR...")
        qwen_result = self.qwen.process(document_result)

        if qwen_result.status != DocumentStatus.SUCCESS:
            logger.warning("Qwen falló por completo, intentando solo con Gemini Vision...")
            return self.gemini.process(document_result)

        if ocr_confidence < self.OCR_CONFIDENCE_THRESHOLD:
            logger.info(f"OCR confidence baja ({ocr_confidence:.2f}). Activando Gemini Vision...")
            gemini_result = self.gemini.process(document_result)

            if gemini_result.status != DocumentStatus.SUCCESS:
                logger.warning("Gemini Vision falló, usando solo resultado de Qwen")
                qwen_result.status = DocumentStatus.SUCCESS
                qwen_result.message = "Estructurado con Qwen (Gemini fallback falló)"
                qwen_result.payload["llm_used"] = "qwen_only_fallback"
                return qwen_result

            qwen_data = qwen_result.payload.get("estructura", {})
            gemini_data = gemini_result.payload.get("estructura", {})

            merged, overwritten = self._merge_results(qwen_data, gemini_data)

            final_result = DocumentResult(
                source_path=document_result.source_path,
                status=DocumentStatus.SUCCESS,
                message="Estructurado exitosamente con estrategia híbrida",
                payload=copy.deepcopy(qwen_result.payload),
            )
            final_result.payload["estructura"] = merged
            final_result.payload["llm_used"] = "qwen_gemini_hybrid"
            final_result.payload["merge_details"] = {
                "qwen_confidence": qwen_data.get("confianza"),
                "gemini_confidence": gemini_data.get("confianza"),
                "ocr_confidence": ocr_confidence,
                "fields_from_gemini": overwritten,
            }

            # CUFE y enlace DIAN desde Gemini (lee el codigo impreso con precision)
            cufe = gemini_data.get("cufe")
            if cufe:
                final_result.payload["cufe"] = cufe
                final_result.payload["cufe_fuente"] = "gemini_vision"
                final_result.payload["dian_url"] = DIAN_QR_URL + str(cufe)

            logger.info(f"Estrategia híbrida completada. Campos corregidos por Gemini: {overwritten}")
            return final_result

        logger.info(f"OCR confidence aceptable ({ocr_confidence:.2f}), usando solo Qwen")
        qwen_result.payload["llm_used"] = "qwen_only"
        return qwen_result

    def _merge_results(self, qwen_data: Dict[str, Any], gemini_data: Dict[str, Any]):
        """Gemini gana en todo campo no nulo que difiera; Qwen aporta el resto."""
        merged = dict(qwen_data)
        overwritten = []

        for field, value in gemini_data.items():
            if field == "observaciones":
                continue
            if value is not None and value != merged.get(field):
                overwritten.append(field)
                merged[field] = value
            elif value is not None and field not in merged:
                merged[field] = value

        q_obs = qwen_data.get("observaciones") or []
        g_obs = gemini_data.get("observaciones") or []
        merged["observaciones"] = list(dict.fromkeys(list(q_obs) + list(g_obs)))

        return merged, overwritten