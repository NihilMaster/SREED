from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

from src.domain.models import RetrievedDocument


class VectorStore(Protocol):
    """
    Puerto para bases de datos vectoriales.

    Implementaciones posibles:
    - ChromaVectorStoreAdapter
    - PgVectorAdapter en el futuro
    - QdrantAdapter en el futuro
    - FAISSAdapter en el futuro

    La aplicacion no debe depender directamente de ChromaDB.
    """

    def initialize(self) -> None:
        """
        Prepara recursos internos:
        - cliente
        - coleccion
        - persistencia
        """
        ...

    def upsert_document(
        self,
        document_id: str,
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Inserta o actualiza un documento vectorizado.

        document_id:
            Identificador estable del documento.
        text:
            Texto que se va a vectorizar.
        metadata:
            Metadata filtrable. Debe mantenerse simple:
            strings, numeros o booleanos.
        """
        ...

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedDocument]:
        """
        Busca documentos similares a la pregunta.

        Debe devolver modelos de dominio RetrievedDocument,
        no objetos internos de ChromaDB.
        """
        ...

    def count(self) -> int:
        """
        Devuelve la cantidad de documentos indexados.
        """
        ...