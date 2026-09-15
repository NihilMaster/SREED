from __future__ import annotations

import os

# Se desactiva la telemetria de ChromaDB antes de importar chromadb.
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import json
import logging
from typing import Any, Dict, List, Optional

import chromadb

from src.domain.models import RetrievedDocument
from src.infrastructure.config import Settings

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Adaptador de ChromaDB para SREED.

    Implementa el puerto VectorStore.

    Este adaptador no debe contener reglas de negocio.
    Solo traduce entre el dominio y ChromaDB.

    PENDIENTE IA / RAG:
    - Aqui se guardaran documentos procesados.
    - El texto vectorizado puede venir del JSON estructurado,
      del OCR crudo o de una combinacion de ambos.
    - Cuando exista OCR real, el texto a vectorizar debera ser
      construido por un caso de uso, no por este adaptador.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Optional[chromadb.api.ClientAPI] = None
        self._collection = None

    def initialize(self) -> None:
        """
        Inicializa cliente y coleccion de ChromaDB.
        """

        if self._collection is not None:
            return

        try:
            self._settings.chroma_dir.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=str(self._settings.chroma_dir),
            )

            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                "ChromaDB inicializado. Coleccion: %s. Ruta: %s",
                self._settings.chroma_collection,
                self._settings.chroma_dir,
            )

        except Exception:
            logger.exception("Error inicializando ChromaDB.")
            raise

    def upsert_document(
        self,
        document_id: str,
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Inserta o actualiza un documento en la coleccion vectorial.
        """

        self._ensure_initialized()

        if not document_id:
            raise ValueError("document_id no puede ser vacio.")

        if not text.strip():
            raise ValueError("text no puede ser vacio al vectorizar.")

        sanitized_metadata = self._sanitize_metadata(metadata)
        sanitized_metadata["document_id"] = document_id

        try:
            self._collection.upsert(
                ids=[document_id],
                documents=[text],
                metadatas=[sanitized_metadata],
            )

            logger.info(
                "Documento indexado en ChromaDB: %s",
                document_id,
            )

        except Exception:
            logger.exception(
                "Error guardando documento en ChromaDB: %s",
                document_id,
            )
            raise

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedDocument]:
        """
        Busca documentos similares.

        Devuelve modelos de dominio RetrievedDocument.
        """

        self._ensure_initialized()

        if not query.strip():
            return []

        if top_k <= 0:
            return []

        try:
            response = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where if where else None,
                include=["documents", "metadatas", "distances"],
            )

            return self._to_retrieved_documents(response)

        except Exception:
            logger.exception("Error consultando ChromaDB.")
            raise

    def count(self) -> int:
        """
        Cantidad de documentos en la coleccion.
        """

        self._ensure_initialized()

        try:
            return int(self._collection.count())
        except Exception:
            logger.exception("Error contando documentos en ChromaDB.")
            raise

    def _ensure_initialized(self) -> None:
        if self._collection is None:
            self.initialize()

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        ChromaDB trabaja mejor con metadata simple.

        Se acepta:
        - str
        - int
        - float
        - bool

        Otros tipos se convierten a string o JSON serializado.
        """

        sanitized: Dict[str, Any] = {}

        if not metadata:
            return sanitized

        for key, value in metadata.items():
            key_str = str(key)

            if value is None:
                sanitized[key_str] = ""
            elif isinstance(value, bool):
                sanitized[key_str] = value
            elif isinstance(value, (str, int, float)):
                sanitized[key_str] = value
            elif isinstance(value, (dict, list)):
                sanitized[key_str] = json.dumps(value, ensure_ascii=False)
            else:
                sanitized[key_str] = str(value)

        return sanitized

    def _to_retrieved_documents(self, response: Any) -> List[RetrievedDocument]:
        """
        Convierte la respuesta cruda de ChromaDB a modelos de dominio.
        """

        if not response:
            return []

        ids = response.get("ids", [[]])[0]
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        retrieved: List[RetrievedDocument] = []

        for index, document_id in enumerate(ids):
            text = documents[index] if index < len(documents) else ""
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else None

            if not isinstance(metadata, dict):
                metadata = {}

            source_file = str(metadata.get("source_file", ""))

            retrieved.append(
                RetrievedDocument(
                    document_id=str(document_id),
                    source_file=source_file,
                    score=self._to_score(distance),
                    snippet=(text or "")[:500],
                    metadata=metadata,
                )
            )

        return retrieved

    def _to_score(self, distance: Any) -> Optional[float]:
        """
        Convierte distancia a score.

        Se usa espacio cosine.
        En cosine, menor distancia significa mayor similitud.
        """

        if distance is None:
            return None

        try:
            distance_value = float(distance)
        except (TypeError, ValueError):
            return None

        score = 1.0 - distance_value

        if score < 0.0:
            score = 0.0

        if score > 1.0:
            score = 1.0

        return score