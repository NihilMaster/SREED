from __future__ import annotations

from typing import List, Protocol

from src.domain.models import RAGAnswer, RetrievedDocument


class LLMProvider(Protocol):
    """
    Puerto para proveedores de respuesta RAG.

    Implementaciones previstas:
    - MockLLMProvider
    - QwenLLMProvider
    - QwenGeminiStrategy

    Regla de negocio:
    - Nunca debe existir un proveedor Gemini solo.
    - Gemini solo puede aparecer como apoyo dentro de una estrategia
      donde Qwen sea el modelo principal.
    """

    def answer(
        self,
        question: str,
        retrieved_documents: List[RetrievedDocument],
    ) -> RAGAnswer:
        """
        Genera una respuesta a partir de una pregunta y documentos recuperados.

        PENDIENTE IA:
        - MockLLMProvider devolvera una respuesta simulada.
        - QwenLLMProvider usara Ollama/Qwen local.
        - QwenGeminiStrategy combinara Qwen local con apoyo opcional de Gemini,
          pero sin convertir a Gemini en proveedor unico.
        """
        ...