from typing import Dict, Any
import logging
import json
from pathlib import Path

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
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-flash-latest",  # Forzado al nombre que funcionó en tu curl
    ):
        if genai is None:
            raise ImportError("Instala: pip install google-genai")

        self.api_key = api_key
        self.model = model
        self.client = genai.Client(api_key=api_key)
        logger.info(f"GeminiVisionProvider inicializado con modelo: {self.model}")

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

    def _extract_from_image(self, image_path: str) -> Dict[str, Any]:
        img_data = Path(image_path).read_bytes()
        ext = Path(image_path).suffix.lower()
        mime_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

        prompt = """You are an invoice data extraction expert. Analyze this invoice image and extract the following fields as JSON:
{
  "proveedor": "Commercial name of the company",
  "nit": "Tax ID number",
  "fecha": "Emission date in YYYY-MM-DD format",
  "numero_factura": "Invoice number",
  "subtotal": number (before taxes),
  "impuestos": number (taxes),
  "total": number (total amount before retentions),
  "moneda": "Currency code (COP, USD, etc) or null",
  "items": [{"descripcion": string, "cantidad": number, "precio_unitario": number, "total_item": number}],
  "confianza": number between 0 and 1,
  "observaciones": ["list of observations"]
}
RULES: Return ONLY valid JSON. For numbers, remove thousand separators. For total, use the value labeled TOTAL / VALOR PAGADO."""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=img_data, mime_type=mime_type)
                ],
                config={
                    "temperature": 0.1,
                    "max_output_tokens": 1000,
                }
            )
            
            response_text = response.text.strip()
            if response_text.startswith("```"):
                response_text = "\n".join(response_text.split("\n")[1:-1])

            structured_data = json.loads(response_text)
            for field in ["proveedor", "nit", "fecha", "total", "confianza"]:
                if field not in structured_data:
                    structured_data[field] = None
            return structured_data
        except Exception as e:
            logger.error(f"Error llamando a Gemini: {e}")
            raise