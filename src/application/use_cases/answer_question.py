from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.application.ports.llm_provider import LLMProvider
from src.application.ports.vector_store import VectorStore
from src.domain.models import RAGAnswer

logger = logging.getLogger(__name__)


class AnswerQuestionUseCase:
    """
    Caso de uso para responder una pregunta usando RAG.

    Flujo:
    1. Recibe una pregunta.
    2. Consulta el VectorStore.
    3. Recupera documentos relevantes.
    4. Delega la respuesta al LLMProvider.

    El caso de uso no sabe si el proveedor es:
    - MockLLMProvider
    - QwenLLMProvider
    - QwenGeminiStrategy

    REGLA:
    - Nunca debe existir un proveedor Gemini solo.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        top_k: int = 5,
    ) -> None:
        self._vector_store = vector_store
        self._llm_provider = llm_provider
        self._top_k = max(1, int(top_k))

    def execute(
        self,
        question: str,
        where: Optional[Dict[str, Any]] = None,
    ) -> RAGAnswer:
        """
        Ejecuta la consulta RAG.

        PENDIENTE IA:
        - Cuando exista Qwen local, el LLMProvider sera QwenLLMProvider.
        - Cuando exista modo hibrido, el LLMProvider sera QwenGeminiStrategy.
        """

        question = question.strip()

        if not question:
            return RAGAnswer(
                question="",
                answer="La pregunta esta vacia.",
                provider="system",
                retrieved_documents=[],
            )

        try:
            retrieved_documents = self._vector_store.search(
                query=question,
                top_k=self._top_k,
                where=where,
            )
        except Exception:
            logger.exception(
                "Error recuperando documentos para la pregunta RAG."
            )
            raise

        try:
            return self._llm_provider.answer(
                question=question,
                retrieved_documents=retrieved_documents,
            )
        except Exception:
            logger.exception(
                "Error generando respuesta RAG con el proveedor actual."
            )
            raise