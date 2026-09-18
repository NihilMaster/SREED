from typing import Dict, Any
import logging
import json
import re
from pathlib import Path

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

from src.application.ports.llm_provider import LLMProvider
from src.domain.models import DocumentResult, DocumentStatus

logger = logging.getLogger(__name__)


class GeminiVisionProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-3.6-flash"):
        if genai is None:
            raise ImportError("Instala: pip install google-genai")

        # Silenciar el warning cosmético de AFC del SDK
        logging.getLogger("google_genai.models").setLevel(logging.ERROR)

        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=api_key)
        logger.info(f"GeminiVisionProvider inicializado con modelo: {model}")

    def process(self, document_result: DocumentResult) -> DocumentResult:
        source_path = document_result.source_path
        if not Path(source_path).exists():
            return DocumentResult(
                source_path=source_path,
                status=DocumentStatus.FAILED,
                message="Imagen no encontrada",
                payload=document_result.payload.copy(),
            )

        try:
            logger.info(f"Extrayendo con Gemini Vision: {source_path}")
            structured_data = self._extract_from_image(str(source_path))

            success_result = DocumentResult(
                source_path=source_path,
                status=DocumentStatus.SUCCESS,
                message="Estructurado con Gemini Vision",
                payload=document_result.payload.copy(),
            )
            success_result.payload["estructura"] = structured_data
            success_result.payload["llm_used"] = "gemini_vision"
            return success_result
        except Exception as e:
            logger.error(f"Error con Gemini Vision: {e}")
            return DocumentResult(
                source_path=source_path,
                status=DocumentStatus.FAILED,
                message=f"Error Gemini: {str(e)}",
                payload=document_result.payload.copy(),
            )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)

    @retry(
        stop=stop_after_attempt(4),  # Aumentar de 3 a 4 intentos
        wait=wait_exponential(multiplier=2, min=3, max=30),  # Esperar más entre reintentos
        retry=retry_if_exception_type((Exception,)),  # Reintentar ante cualquier error
        reraise=True
    )
    def _extract_from_image(self, image_path: str) -> Dict[str, Any]:
        img_data = Path(image_path).read_bytes()
        ext = Path(image_path).suffix.lower()
        mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

        prompt = """You are an invoice data extraction expert. Analyze this invoice image and return ONLY a compact JSON object with exactly these keys:
{"proveedor": string|null, "nit": string|null, "fecha": "YYYY-MM-DD"|null, "numero_factura": string|null, "subtotal": number|null, "impuestos": number|null, "total": number|null, "moneda": string|null, "cufe": string|null, "items": [{"descripcion": string, "cantidad": number, "precio_unitario": number, "total_item": number}], "confianza": number, "observaciones": [string]}
Rules:
1. Numbers without thousand separators ("300.000" -> 300000).
2. total = value labeled TOTAL / TOTAL SIN RETENCIONES (before retentions), NOT "total con retenciones".
3. cufe = the long alphanumeric CUFE code printed below the QR code, transcribed exactly. If absent, null.
4. Item descriptions: max 80 characters. Include at most 10 items.
5. If a field is not legible, use null. Never invent data.
6. No markdown, no explanations."""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=img_data, mime_type=mime_type)
                ],
                config={
                    "temperature": 0.1,
                    "max_output_tokens": 4096,  # Antes 1000: causaba JSON truncado
                }
            )

            response_text = response.text.strip()

            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)

            try:
                structured_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON inválido de Gemini. Respuesta cruda: {response_text[:500]}...")
                raise ValueError(f"Gemini devolvió JSON inválido: {e}")

            for field in ["proveedor", "nit", "fecha", "total", "confianza"]:
                if field not in structured_data:
                    structured_data[field] = None
            return structured_data

        except Exception as e:
            logger.error(f"Error llamando a Gemini: {e}")
            raise