from typing import Dict, Any
import logging
import json
import time
from ollama import Client
from tenacity import retry, stop_after_attempt, wait_exponential

from src.application.ports.llm_provider import LLMProvider
from src.domain.models import DocumentResult, DocumentStatus

logger = logging.getLogger(__name__)


# Tarea corta: solo campos clave. Los items y el CUFE los aporta Gemini Vision.
PROMPT_EXTRACCION_COMPACTO = """You are an invoice data extraction engine. From the OCR text, return ONLY a compact JSON object with exactly these keys and nothing else:
{{"proveedor": string|null, "nit": string|null, "fecha": "YYYY-MM-DD"|null, "numero_factura": string|null, "subtotal": number|null, "impuestos": number|null, "total": number|null, "moneda": string|null, "confianza": number}}
Rules:
1. Numbers without thousand separators ("300.000" -> 300000, "1,500.00" -> 1500.00).
2. total = value labeled TOTAL / VALOR PAGADO / TOTAL SIN RETENCIONES (before retentions).
3. If a field is not legible or absent, use null. Never invent data.
4. No explanations, no markdown, no extra keys.
OCR TEXT:
{raw_text}
JSON:"""


class QwenOllamaProvider(LLMProvider):
    """
    Proveedor Qwen local con salida JSON garantizada por gramatica (format="json")
    y modo de pensamiento desactivado para evitar respuestas vacias.
    """

    def __init__(
        self,
        primary_model: str = "qwen3.5:9b",
        fallback_model: str = "qwen2.5:7b",
        host: str = "http://localhost:11434",
        timeout: int = 1800,
        context_window: int = 4096,
    ):
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.host = host
        self.timeout = timeout
        self.context_window = context_window

        self.client = Client(host=host, timeout=self.timeout)
        self._primary_failures = 0
        self._primary_disabled = False

        logger.info(
            f"QwenOllamaProvider inicializado. Principal: {primary_model}, "
            f"Fallback: {fallback_model}, Timeout: {timeout}s"
        )

    def process(self, document_result: DocumentResult) -> DocumentResult:
        raw_text = document_result.payload.get("raw_text", "")
        if not raw_text.strip():
            document_result.status = DocumentStatus.FAILED
            document_result.message = "No se extrajo texto del documento"
            return document_result

        try:
            structured_data = self._extract_with_fallback(raw_text)
            document_result.payload["estructura"] = structured_data
            document_result.payload["llm_used"] = structured_data.pop("_llm_used", "unknown")
            document_result.status = DocumentStatus.SUCCESS
            document_result.message = "Estructurado exitosamente con Qwen"
            return document_result
        except Exception as e:
            logger.error(f"Error estructurando con Qwen: {e}")
            document_result.status = DocumentStatus.FAILED
            document_result.message = f"Error LLM: {str(e)}"
            return document_result

    def _extract_with_fallback(self, raw_text: str) -> Dict[str, Any]:
        if not self._primary_disabled:
            try:
                logger.info(f"Intentando con modelo principal: {self.primary_model} (Timeout: {self.timeout}s)")
                result = self._call_qwen(self.primary_model, raw_text)
                result["_llm_used"] = self.primary_model
                self._primary_failures = 0
                return result
            except Exception as e:
                self._primary_failures += 1
                logger.warning(f"Modelo principal {self.primary_model} fallo ({self._primary_failures}/3): {e}")
                if self._primary_failures >= 3:
                    logger.error("Modelo principal deshabilitado por fallos consecutivos.")
                    self._primary_disabled = True

        logger.info(f"Activando fallback: {self.fallback_model}")
        result = self._call_qwen(self.fallback_model, raw_text)
        result["_llm_used"] = self.fallback_model
        return result

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    def _call_qwen(self, model: str, raw_text: str) -> Dict[str, Any]:
        max_chars = (self.context_window - 500) * 4
        if len(raw_text) > max_chars:
            logger.warning(f"Texto largo ({len(raw_text)} chars), truncando a {max_chars}")
            raw_text = raw_text[:max_chars] + "\n\n[TEXTO TRUNCADO]"

        prompt = PROMPT_EXTRACCION_COMPACTO.format(raw_text=raw_text)
        options = {"temperature": 0.0, "num_ctx": self.context_window, "num_predict": 768}

        start = time.time()
        try:
            # format="json" garantiza JSON valido por gramatica.
            # think=False apaga el modo razonamiento de Qwen3.x (respuestas vacias).
            try:
                response = self.client.generate(
                    model=model, prompt=prompt, format="json",
                    think=False, options=options, stream=False
                )
            except TypeError:
                # Versiones antiguas de ollama sin parametro think
                response = self.client.generate(
                    model=model, prompt=prompt, format="json",
                    options=options, stream=False
                )
            elapsed = time.time() - start
            logger.info(f"LLM {model} respondio en {elapsed:.1f}s")

            response_text = response.get("response", "").strip()
            if response_text.startswith("```"):
                response_text = "\n".join(response_text.split("\n")[1:-1])

            try:
                structured_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON inválido de {model}. Respuesta cruda: {response_text[:500]}...")
                raise ValueError(f"El modelo devolvió JSON inválido: {e}")

            for field in ["proveedor", "nit", "fecha", "total", "confianza"]:
                if field not in structured_data:
                    structured_data[field] = None

            return structured_data

        except Exception as e:
            logger.error(f"Error inesperado llamando a {model}: {e}")
            raise