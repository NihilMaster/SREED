from typing import Dict, Any
import logging
import json
import time
from ollama import Client
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.application.ports.llm_provider import LLMProvider
from src.domain.models import DocumentResult, DocumentStatus

logger = logging.getLogger(__name__)


PROMPT_EXTRACCION = """Eres un motor de extraccion de datos de facturas.
Recibiras texto OCR posiblemente ruidoso. Devuelve UNICAMENTE un JSON valido, sin markdown, con esta estructura:

{{
  "proveedor": string|null,
  "nit": string|null,
  "fecha": "YYYY-MM-DD"|null,
  "numero_factura": string|null,
  "subtotal": number|null,
  "impuestos": number|null,
  "total": number|null,
  "moneda": string|null,
  "items": [{{"descripcion": string, "cantidad": number, "precio_unitario": number, "total_item": number}}],
  "confianza": number entre 0 y 1,
  "observaciones": [string]
}}

REGLAS DE ORO:
1. NO inventes nada. Si un dato no aparece literalmente en el texto o no puede derivarse por suma simple, usa null y agregalo a "observaciones".
2. Numeros: devuelve siempre numero JSON (nunca string). Normaliza el formato local:
   - Formato colombiano/latam: "69.990" = 69990 ; "1.234,56" = 1234.56 (punto=miles, coma=decimales).
   - Formato anglo: "1,500.00" = 1500.00 (coma=miles, punto=decimales).
   Decide segun el idioma y patron del documento.
3. total: usa el valor etiquetado como TOTAL / TOTAL A PAGAR / VALOR PAGADO. Verifica que subtotal+impuestos sea aproximadamente total; si no cuadra, usa el TOTAL etiquetado y explicalo en "observaciones".
4. fecha: fecha de EMISION de la factura (no vencimiento). Acepta formatos como "2024-04-22", "24/11/20", "17 DE MARZO DE 2020" y normaliza a YYYY-MM-DD. Si hay varias fechas, elige la de emision o venta.
5. moneda: solo si aparece simbolo o codigo explicito (COP, USD, EUR, etc.). Si solo aparece "$" sin pais, usa null y anotalo en "observaciones". Nunca asumas una moneda por defecto.
6. numero_factura: usa el consecutivo de factura (ej. "F693 286219", "INV-1001"). No uses codigos de autorizacion, NIT ni numeros de tarjeta.
7. items: incluye solo lineas con descripcion y al menos precio legible. Ignora texto suelto sin sentido.
8. confianza calibrada: 0.8-1.0 solo si proveedor, total y fecha son legibles y la matematica cuadra; 0.5-0.7 si hay campos clave legibles con dudas; menor a 0.5 si el texto esta muy corrupto.
9. Si el texto parece contener dos documentos o columnas duplicadas, extrae solo el documento principal y anotalo en "observaciones".

TEXTO OCR:
{raw_text}

JSON:"""


class QwenOllamaProvider(LLMProvider):
    """
    Proveedor LLM usando Qwen local via Ollama.

    Estrategia de fallback y tolerancia a fallos:
    1. Modelo principal (ej. qwen3.5-9b-4k-OCR:latest)
    2. Si el principal acumula 3 fallos consecutivos, se deshabilita temporalmente.
    3. Modelo fallback (ej. qwen2.5:7b)
    """

    def __init__(
        self,
        # primary_model: str = "qwen3.5-9b-4k-OCR:latest",
        primary_model: str = "qwen3.5:9b",
        fallback_model: str = "qwen2.5:7b",
        host: str = "http://localhost:11434",
        timeout: int = 1800,  # Timeout alto para modelos de razonamiento
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
            f"QwenOllamaProvider inicializado. "
            f"Principal: {primary_model}, Fallback: {fallback_model}, Timeout: {timeout}s"
        )

    def process(self, document_result: DocumentResult) -> DocumentResult:
        """
        Procesa el resultado OCR y extrae datos estructurados usando Qwen.
        Lee y escribe dentro del diccionario `payload`.
        """
        raw_text = document_result.payload.get("raw_text", "")

        if not raw_text.strip():
            logger.warning("Documento sin texto, no se puede estructurar")
            document_result.status = DocumentStatus.FAILED
            document_result.message = "No se extrajo texto del documento"
            return document_result

        try:
            structured_data = self._extract_with_fallback(raw_text)

            document_result.payload["estructura"] = structured_data
            document_result.payload["llm_used"] = structured_data.pop("_llm_used", "unknown")
            document_result.status = DocumentStatus.SUCCESS
            document_result.message = "Estructurado exitosamente con Qwen"

            logger.info("Documento estructurado exitosamente con Qwen")
            return document_result

        except Exception as e:
            logger.error(f"Error estructurando documento con Qwen: {e}")
            document_result.status = DocumentStatus.FAILED
            document_result.message = f"Error al estructurar con LLM: {str(e)}"
            return document_result

    def _extract_with_fallback(self, raw_text: str) -> Dict[str, Any]:
        """Intenta extraer datos con fallback entre modelos y deshabilitación tras fallos repetidos."""
        if not self._primary_disabled:
            try:
                logger.info(f"Intentando con modelo principal: {self.primary_model} (Timeout: {self.timeout}s)")
                result = self._call_qwen(self.primary_model, raw_text)
                result["_llm_used"] = self.primary_model
                self._primary_failures = 0
                return result
            except Exception as e:
                self._primary_failures += 1
                logger.warning(
                    f"Modelo principal {self.primary_model} fallo ({self._primary_failures}/3): {e}."
                )
                if self._primary_failures >= 3:
                    logger.error("Modelo principal deshabilitado temporalmente por fallos consecutivos.")
                    self._primary_disabled = True

        try:
            logger.info(f"Activando fallback: {self.fallback_model}")
            result = self._call_qwen(self.fallback_model, raw_text)
            result["_llm_used"] = self.fallback_model
            return result
        except Exception as e:
            logger.error(f"Ambos modelos Qwen fallaron: {e}")
            raise RuntimeError(f"No se pudo estructurar el documento: {e}")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _call_qwen(self, model: str, raw_text: str) -> Dict[str, Any]:
        # 1 token ≈ 4 caracteres. Dejamos 500 tokens de margen.
        max_chars = (self.context_window - 500) * 4
        if len(raw_text) > max_chars:
            logger.warning(
                f"Texto muy largo ({len(raw_text)} chars), truncando a {max_chars} chars"
            )
            raw_text = raw_text[:max_chars] + "\n\n[TEXTO TRUNCADO]"

        prompt = PROMPT_EXTRACCION.format(raw_text=raw_text)
        start_time = time.time()

        try:
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "num_ctx": self.context_window,
                    "num_predict": 1000,
                },
                stream=False,
            )

            elapsed = time.time() - start_time
            logger.info(f"LLM {model} respondio en {elapsed:.1f}s")

            response_text = response.get("response", "").strip()

            # Limpiar bloques de código Markdown sobrantes si existen
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            try:
                structured_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Respuesta no es JSON valido de {model}: {response_text[:200]}...")
                raise ValueError(f"El modelo devolvio JSON invalido: {e}")

            required_fields = ["proveedor", "nit", "fecha", "total", "confianza"]
            for field in required_fields:
                if field not in structured_data:
                    structured_data[field] = None

            logger.info(
                f"Extraccion exitosa con {model}. "
                f"Confianza: {structured_data.get('confianza', 'N/A')}"
            )

            return structured_data

        except TimeoutError:
            logger.error(f"Timeout llamando a {model}")
            raise
        except ConnectionError:
            logger.error(f"Error de conexion con Ollama en {self.host}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado llamando a {model}: {e}")
            raise

    def answer_question(self, question: str, context: str) -> str:
        """Responde una pregunta usando el contexto proporcionado (para RAG)."""
        prompt = f"""Eres un asistente experto en análisis de facturas y documentos financieros.
Responde la siguiente pregunta basándote ÚNICAMENTE en el contexto proporcionado.
Si la información no está en el contexto, indica que no tienes suficiente información.

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA:"""

        try:
            response = self.client.generate(
                model=self.primary_model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "num_ctx": self.context_window,
                },
                stream=False,
            )
            return response.get("response", "No se pudo generar respuesta")

        except Exception as e:
            logger.error(f"Error respondiendo pregunta: {e}")
            return f"Error al procesar la pregunta: {str(e)}"