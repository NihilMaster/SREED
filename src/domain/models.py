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
    Resultado canónico del procesamiento de un documento.
    Usa Pydantic para validación y serialización automática a JSON.
    """
    document_id: str = Field(default_factory=lambda: uuid4().hex)
    source_path: Path
    final_path: Optional[Path] = None
    status: DocumentStatus = DocumentStatus.PENDING
    message: str = ""
    
    # Payload flexible para guardar OCR crudo, QR, y datos estructurados
    payload: Dict[str, Any] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        # Permite que Path se serialice correctamente a string en JSON
        json_encoders = {
            Path: str
        }


class ChatRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Mensaje de chat entre usuario y asistente."""
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
    """
    question: str
    answer: str
    provider: str
    retrieved_documents: List[RetrievedDocument] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))