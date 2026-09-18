from typing import Dict, Any
import logging
import json
import time
from pathlib import Path
from ollama import Client
from tenacity import retry, stop_after_attempt, wait_exponential

from src.application.ports.llm_provider import LLMProvider
from src.domain.models import DocumentResult, DocumentStatus

logger = logging.getLogger(__name__)
rag_logger = logging.getLogger("sreed.rag")


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


PROMPT_RAG = """You are an analytical assistant for processed invoices.
Answer the question using ONLY the information inside <context>.
If the answer is not present in <context>, reply exactly:
"No tengo informacion suficiente en los documentos proporcionados para responder esta pregunta."
For each important statement, cite the source as [Doc N], where N is the document number in <context>.
Keep the answer concise (max 200 tokens).

<context>
{context}
</context>

Question: {question}

Answer:"""


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

    # ---------- Estructuracion ----------

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
        options = {
            "temperature": 0.0,
            "num_ctx": self.context_window,
            "num_predict": 768,
            "keep_alive": "30m",  # Mantener modelo en memoria
        }

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

    # ---------- RAG ----------

    def answer_question(self, question: str, context: str) -> str:
        """
        Responde preguntas usando RAG con timeout extendido.
        Siempre usa el fallback (qwen2.5:7b) porque es más estable para texto libre.
        Qwen 3.5 está optimizado para extracción JSON, no para generación libre.
        """
        from tenacity import retry, stop_after_attempt, wait_exponential
        
        logger.info(f"Pregunta RAG recibida: {question}")
        logger.info(f"Contexto: {len(context)} caracteres")
        
        # Timeout específico para RAG (más generoso que extracción)
        rag_timeout = 300  # 5 minutos
        
        @retry(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True
        )
        def _call_with_timeout(model: str, prompt: str, opts: dict) -> str:
            try:
                client = Client(host=self.host, timeout=rag_timeout)
                
                # Para RAG NO usamos think=False ni format="json"
                # Queremos generación libre de texto
                response = client.generate(
                    model=model,
                    prompt=prompt,
                    options=opts,
                    stream=False
                )
                
                if not response or "response" not in response:
                    raise ValueError("Respuesta vacía del modelo")
                
                # Log crudo antes de limpiar para diagnóstico
                raw_response = response["response"]
                logger.debug(f"RAG respuesta cruda ({len(raw_response)} chars): {raw_response[:200]}...")
                
                answer = raw_response.strip()
                
                if not answer:
                    raise ValueError(f"Respuesta vacía después de strip. Crudo: {raw_response[:100]}")
                
                logger.info(f"RAG exitoso con {model}: {len(answer)} caracteres")
                return answer
                
            except Exception as e:
                logger.error(f"Error en RAG con {model}: {e}")
                raise
        
        prompt = self._build_rag_prompt(question, context)
        
        # Opciones para generación libre (sin restricciones JSON)
        options = {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 1024,
            "num_ctx": self.context_window
        }
        
        # Siempre usar el fallback para RAG (más estable para texto libre)
        logger.info(f"Intentando RAG con {self.fallback_model} (optimizado para texto libre)")
        return _call_with_timeout(self.fallback_model, prompt, options)

    def _build_rag_prompt(self, question: str, context: str) -> str:
            """
            Construye el prompt para RAG con validaciones de longitud.
            
            Args:
                question: Pregunta del usuario
                context: Contexto recuperado de FAISS
                
            Returns:
                Prompt formateado para el modelo
            """
            # Limitar contexto a ~3000 tokens (aprox 12000 caracteres)
            max_context_chars = 12000
            if len(context) > max_context_chars:
                logger.warning(f"Contexto muy largo ({len(context)} chars), truncando a {max_context_chars}")
                context = context[:max_context_chars] + "\n[...CONTEXTO TRUNCADO...]"
            
            prompt = f"""Basándote ÚNICAMENTE en el siguiente contexto, responde la pregunta del usuario.
    
    CONTEXTO:
    {context}
    
    PREGUNTA: {question}
    
    INSTRUCCIONES:
    - Responde en el mismo idioma de la pregunta
    - Si la información no está en el contexto, di: "No tengo información suficiente en los documentos para responder esta pregunta"
    - Sé conciso y directo
    - Si mencionas datos específicos (números, fechas, nombres), cita el documento fuente entre paréntesis
    - Usa formato markdown para mejor legibilidad
    
    RESPUESTA:"""
            
            return prompt

    def _call_rag(self, model: str, prompt: str, options: Dict[str, Any]) -> str:
        start = time.time()
        try:
            try:
                response = self.client.generate(
                    model=model, prompt=prompt,
                    think=False, options=options, stream=False
                )
            except TypeError:
                response = self.client.generate(
                    model=model, prompt=prompt, options=options, stream=False
                )
            
            elapsed = time.time() - start
            text = (response.get("response") or "").strip()
            
            if not text:
                rag_logger.error(f"Respuesta vacia del modelo {model} en RAG")
                raise ValueError("Respuesta vacia del modelo en RAG")
            
            rag_logger.info(f"RAG respondido por {model} en {elapsed:.1f}s ({len(text)} chars)")
            return text
            
        except Exception as e:
            elapsed = time.time() - start
            rag_logger.error(f"Error RAG llamando a {model} despues de {elapsed:.1f}s: {e}")
            raise