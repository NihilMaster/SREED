from typing import Dict, Any
import logging
import copy

from src.application.ports.llm_provider import LLMProvider
from src.domain.models import DocumentResult, DocumentStatus
from src.infrastructure.adapters.llm.qwen_ollama_provider import QwenOllamaProvider
from src.infrastructure.adapters.llm.gemini_vision_provider import GeminiVisionProvider

logger = logging.getLogger(__name__)


class QwenGeminiStrategy(LLMProvider):
    """
    Estrategia híbrida que combina Qwen (local) + Gemini Vision (nube).
    """

    OCR_CONFIDENCE_THRESHOLD = 0.6

    def __init__(
        self,
        qwen_provider: QwenOllamaProvider,
        gemini_provider: GeminiVisionProvider,
    ):
        self.qwen = qwen_provider
        self.gemini = gemini_provider
        logger.info("QwenGeminiStrategy inicializado (híbrido)")

    def process(self, document_result: DocumentResult) -> DocumentResult:
        ocr_confidence = document_result.payload.get("ocr_confidence", 0.0)

        logger.info(
            f"Estrategia híbrida: OCR confidence = {ocr_confidence:.2f}, "
            f"umbral = {self.OCR_CONFIDENCE_THRESHOLD}"
        )

        # 1. Qwen siempre extrae del texto OCR
        logger.info("Paso 1: Qwen extrayendo del texto OCR...")
        qwen_result = self.qwen.process(document_result)

        if qwen_result.status != DocumentStatus.SUCCESS:
            logger.warning("Qwen falló, intentando solo con Gemini Vision...")
            return self.gemini.process(document_result)

        # 2. Si OCR confidence es baja, Gemini Vision valida/extrae
        if ocr_confidence < self.OCR_CONFIDENCE_THRESHOLD:
            logger.info(
                f"OCR confidence baja ({ocr_confidence:.2f} < {self.OCR_CONFIDENCE_THRESHOLD}). "
                f"Activando Gemini Vision..."
            )

            gemini_result = self.gemini.process(document_result)

            if gemini_result.status != DocumentStatus.SUCCESS:
                logger.warning("Gemini Vision falló, usando solo resultado de Qwen")
                # CORRECCIÓN: Aseguramos que el estado de éxito de Qwen se mantenga
                qwen_result.status = DocumentStatus.SUCCESS
                qwen_result.message = "Estructurado con Qwen (Gemini fallback falló)"
                qwen_result.payload["llm_used"] = "qwen_only_fallback"
                return qwen_result

            qwen_data = qwen_result.payload.get("estructura", {})
            gemini_data = gemini_result.payload.get("estructura", {})

            # 3. Merge inteligente
            merged_data = self._merge_results(qwen_data, gemini_data, ocr_confidence)

            # Creamos un nuevo resultado exitoso con los datos fusionados
            final_result = DocumentResult(
                source_path=document_result.source_path,
                status=DocumentStatus.SUCCESS,
                message="Estructurado exitosamente con estrategia híbrida",
                payload=copy.deepcopy(qwen_result.payload),
            )
            final_result.payload["estructura"] = merged_data
            final_result.payload["llm_used"] = "qwen_gemini_hybrid"
            final_result.payload["merge_details"] = {
                "qwen_confidence": qwen_data.get("confianza"),
                "gemini_confidence": gemini_data.get("confianza"),
                "ocr_confidence": ocr_confidence,
                "fields_from_gemini": self._get_changed_fields(qwen_data, merged_data),
            }

            logger.info("Estrategia híbrida completada (Qwen + Gemini Vision)")
            return final_result

        else:
            # OCR fue bueno, solo usar Qwen
            logger.info(f"OCR confidence aceptable ({ocr_confidence:.2f}), usando solo Qwen")
            qwen_result.payload["llm_used"] = "qwen_only"
            return qwen_result

    def _merge_results(
        self,
        qwen_data: Dict[str, Any],
        gemini_data: Dict[str, Any],
        ocr_confidence: float,
    ) -> Dict[str, Any]:
        merged = dict(qwen_data)

        # Campos críticos donde Gemini siempre gana (cuando OCR es malo)
        critical_fields = ["total", "subtotal", "impuestos", "proveedor", "nit"]

        for field in critical_fields:
            gemini_value = gemini_data.get(field)
            if gemini_value is not None:
                old_value = merged.get(field)
                merged[field] = gemini_value
                if old_value != gemini_value:
                    logger.info(
                        f"Campo crítico '{field}': {old_value} → {gemini_value} (Gemini)"
                    )

        # Campos no críticos: usar el que tenga mejor valor
        non_critical_fields = ["fecha", "numero_factura", "moneda"]
        for field in non_critical_fields:
            gemini_value = gemini_data.get(field)
            if gemini_value is not None and merged.get(field) is None:
                merged[field] = gemini_value

        # Items: Gemini gana si OCR es malo
        if ocr_confidence < 0.5:
            gemini_items = gemini_data.get("items", [])
            if gemini_items:
                merged["items"] = gemini_items
                logger.info(f"Items reemplazados por Gemini ({len(gemini_items)} items)")

        # Confianza: usar la de Gemini si OCR es malo
        if ocr_confidence < self.OCR_CONFIDENCE_THRESHOLD:
            merged["confianza"] = gemini_data.get("confianza", 0.7)

        # Observaciones: combinar
        qwen_obs = qwen_data.get("observaciones", [])
        gemini_obs = gemini_data.get("observaciones", [])
        merged["observaciones"] = list(set(qwen_obs + gemini_obs))

        return merged

    def _get_changed_fields(self, original: Dict[str, Any], merged: Dict[str, Any]) -> list:
        changed = []
        for key in original.keys():
            if original[key] != merged.get(key):
                changed.append(key)
        return changed