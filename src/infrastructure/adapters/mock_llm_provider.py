from __future__ import annotations

import logging
from typing import List

from src.domain.models import RAGAnswer, RetrievedDocument

logger = logging.getLogger(__name__)


class MockLLMProvider:
    """
    Proveedor simulado de respuestas RAG.

    No llama a ningun modelo real.

    Su proposito es dejar lista la interfaz y el flujo para que
    posteriormente se conecte:

    - QwenLLMProvider
    - QwenGeminiStrategy

    REGLA:
    - Nunca se debe implementar un proveedor Gemini solo.
    - Gemini solo puede existir como apoyo dentro de una estrategia
      donde Qwen sea el modelo principal.

    PENDIENTE IA:
    - Reemplazar este mock por QwenLLMProvider cuando exista Ollama/Qwen.
    - En modo qwen_gemini, crear estrategia compuesta QwenGeminiStrategy.
    """

    def answer(
        self,
        question: str,
        retrieved_documents: List[RetrievedDocument],
    ) -> RAGAnswer:
        retrieved_documents = retrieved_documents or []

        if not retrieved_documents:
            answer_text = (
                "[Respuesta simulada] No se recuperaron documentos. "
                "Cuando se conecte Qwen local, el modelo respondera "
                "usando el historico de facturas procesadas."
            )
        else:
            files = ", ".join(
                doc.source_file or doc.document_id
                for doc in retrieved_documents[:3]
            )

            answer_text = (
                f"[Respuesta simulada] Se recuperaron "
                f"{len(retrieved_documents)} documentos. "
                f"Documentos principales: {files}. "
                "La respuesta real quedara a cargo de Qwen local "
                "o de una estrategia Qwen + Gemini."
            )

        logger.info(
            "MockLLMProvider respondiendo pregunta: %s",
            question,
        )

        return RAGAnswer(
            question=question,
            answer=answer_text,
            provider="mock",
            retrieved_documents=retrieved_documents,
        )