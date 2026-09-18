from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.application.ports.llm_provider import LLMProvider
from src.application.ports.vector_store import VectorStore
from src.domain.models import RAGAnswer, RetrievedDocument

logger = logging.getLogger(__name__)


class AnswerQuestionUseCase:
    """
    Caso de uso RAG:
    1. Recupera documentos relevantes de la base vectorial.
    2. Construye un contexto numerado con metadatos de cabecera.
    3. Delega la respuesta al LLM local (nunca a Gemini como proveedor unico).
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
        question = question.strip()

        if not question:
            return RAGAnswer(
                question="",
                answer="La pregunta está vacía.",
                provider="system",
                retrieved_documents=[],
            )

        try:
            retrieved = self._vector_store.search(
                query=question, top_k=self._top_k, where=where
            )
        except Exception:
            logger.exception("Error recuperando documentos para RAG.")
            raise

        if not retrieved:
            return RAGAnswer(
                question=question,
                answer="No hay documentos indexados que coincidan con la pregunta.",
                provider="system",
                retrieved_documents=[],
            )

        context = self._build_context(retrieved)

        try:
            answer_text = self._llm_provider.answer_question(question, context)
        except Exception:
            logger.exception("Error generando respuesta RAG.")
            raise

        provider = f"qwen-rag:{getattr(self._llm_provider, 'primary_model', 'local')}"

        return RAGAnswer(
            question=question,
            answer=answer_text,
            provider=provider,
            retrieved_documents=retrieved,
        )

    @staticmethod
    def _build_context(retrieved: List[RetrievedDocument]) -> str:
        blocks = []
        for index, doc in enumerate(retrieved, start=1):
            meta = doc.metadata or {}
            header = (
                f"[Doc {index}] fuente={doc.source_file or doc.document_id} "
                f"proveedor={meta.get('proveedor', 'desconocido')} "
                f"fecha={meta.get('fecha', 'desconocida')} "
                f"total={meta.get('total', 'desconocido')}"
            )
            blocks.append(f"{header}\n{doc.snippet}")
        return "\n\n".join(blocks)