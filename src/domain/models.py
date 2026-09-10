from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    UNSTABLE = "unstable"


class DocumentResult(BaseModel):
    """
    Resultado canonico del procesamiento de un documento.

    Este modelo pertenece al dominio. No debe depender de OpenCV,
    EasyOCR, ChromaDB, Streamlit ni detalles de infraestructura.
    """

    document_id: str = Field(default_factory=lambda: uuid4().hex)
    source_path: Path
    final_path: Optional[Path] = None
    status: DocumentStatus = DocumentStatus.PENDING
    message: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """
    Mensaje de chat entre usuario y asistente.
    """

    role: ChatRole
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RetrievedDocument(BaseModel):
    """
    Documento recuperado desde una base vectorial.

    No depende de ChromaDB. El adaptador de infraestructura debe
    convertir los resultados de ChromaDB a este modelo.
    """

    document_id: str
    source_file: str = ""
    score: Optional[float] = None
    snippet: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGAnswer(BaseModel):
    """
    Respuesta estandarizada del flujo RAG.

    La interfaz Streamlit solo debe consumir este modelo.
    No debe importar si respondio:
    - MockLLMProvider
    - QwenLLMProvider
    - QwenGeminiStrategy

    PENDIENTE IA:
    - Cuando exista Qwen local, este modelo sera llenado por QwenLLMProvider.
    - Cuando exista estrategia hibrida, sera llenado por QwenGeminiStrategy.
    """

    question: str
    answer: str
    provider: str
    retrieved_documents: List[RetrievedDocument] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))