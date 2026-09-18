from typing import Protocol

from src.domain.models import DocumentResult


class LLMProvider(Protocol):
    """
    Puerto para proveedores de estructuracion y respuesta RAG.

    Implementaciones:
    - QwenOllamaProvider (local)
    - QwenGeminiStrategy (hibrida: Qwen principal + Gemini como apoyo)

    Regla de negocio:
    - Gemini nunca actua como proveedor unico.
    """

    def process(self, document_result: DocumentResult) -> DocumentResult:
        """Estructura el texto OCR dentro de payload['estructura']."""
        ...

    def answer_question(self, question: str, context: str) -> str:
        """
        Responde una pregunta usando UNICAMENTE el contexto recuperado.
        Devuelve texto libre (no JSON).
        """
        ...