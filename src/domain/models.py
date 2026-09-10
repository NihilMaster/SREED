from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
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