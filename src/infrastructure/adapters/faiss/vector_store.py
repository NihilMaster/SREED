from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Instala: pip install faiss-cpu sentence-transformers")

from src.application.ports.vector_store import VectorStore
from src.domain.models import RetrievedDocument

logger = logging.getLogger(__name__)


class FAISSVectorStore(VectorStore):
    """Adaptador de FAISS para búsqueda vectorial sin problemas de locks de SQLite."""

    def __init__(self, settings) -> None:
        self._settings = settings
        self._index = None
        self._documents: List[Dict[str, Any]] = []
        # Modelo ligero y rápido (384 dimensiones)
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Usamos el directorio configurado para persistencia
        base_dir = Path(getattr(settings, "faiss_dir", settings.faiss_dir))
        self._index_path = base_dir / "faiss.index"
        self._docs_path = base_dir / "documents.json"

    def initialize(self) -> None:
        """Carga el índice y los documentos desde disco, o crea uno nuevo."""
        base_dir = self._index_path.parent
        base_dir.mkdir(parents=True, exist_ok=True)

        if self._index_path.exists() and self._docs_path.exists():
            logger.info(f"Cargando índice FAISS desde {self._index_path}")
            self._index = faiss.read_index(str(self._index_path))
            with open(self._docs_path, "r", encoding="utf-8") as f:
                self._documents = json.load(f)
            logger.info(f"FAISS inicializado con {len(self._documents)} documentos.")
        else:
            logger.info("Creando nuevo índice FAISS.")
            self._index = faiss.IndexFlatIP(384)  # Inner Product (similitud de coseno si está normalizado)
            self._documents = []
            self._save()

    def _save(self) -> None:
        """Guarda el índice y los metadatos en disco."""
        faiss.write_index(self._index, str(self._index_path))
        with open(self._docs_path, "w", encoding="utf-8") as f:
            json.dump(self._documents, f, ensure_ascii=False, indent=2)

    def upsert_document(self, document_id: str, text: str, metadata: Dict[str, Any]) -> None:
        """Inserta o actualiza un documento."""
        # FAISS no tiene update nativo, así que eliminamos primero si existe
        self.delete_document(document_id)

        embedding = self._model.encode([text])[0]
        # Normalizar para que el producto punto sea equivalente a similitud de coseno
        embedding = embedding / np.linalg.norm(embedding)
        
        self._index.add(np.array([embedding], dtype=np.float32))
        self._documents.append({
            "document_id": document_id,
            "text": text,
            "metadata": metadata
        })
        self._save()

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedDocument]:
        """Busca documentos similares. Filtra post-búsqueda si se usa 'where'."""
        if self._index is None or len(self._documents) == 0:
            return []

        query_embedding = self._model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        k = min(top_k, len(self._documents))
        distances, indices = self._index.search(np.array([query_embedding], dtype=np.float32), k)

        documents: List[RetrievedDocument] = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            
            doc = self._documents[idx]
            doc_meta = doc["metadata"]
            
            # Filtrado post-búsqueda (simulación de 'where')
            if where:
                match = True
                for key, value in where.items():
                    if doc_meta.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            score = float(distances[0][i])
            
            # CORRECCIÓN CLAVE: Mapear 'text' a 'snippet' y exponer 'source_file'
            retrieved_doc = RetrievedDocument(
                document_id=doc["document_id"],
                snippet=doc["text"],  # <-- Aquí estaba el fallo
                metadata=doc_meta,
                score=score,
            )
            
            # Inyección de seguridad por si tu clase RetrievedDocument espera source_file como atributo directo
            if hasattr(retrieved_doc, 'source_file') and not getattr(retrieved_doc, 'source_file', None):
                retrieved_doc.source_file = doc_meta.get("source_file", "")
            
            documents.append(retrieved_doc)
        
        return documents

    def count(self) -> int:
        return len(self._documents)

    def delete_document(self, document_id: str) -> None:
        """Elimina un documento por su ID reconstruyendo el índice."""
        to_keep = [doc for doc in self._documents if doc["document_id"] != document_id]
        if len(to_keep) < len(self._documents):
            self._documents = to_keep
            self._index = faiss.IndexFlatIP(384)
            if self._documents:
                texts = [doc["text"] for doc in self._documents]
                embeddings = self._model.encode(texts)
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                self._index.add(np.array(embeddings, dtype=np.float32))
            self._save()