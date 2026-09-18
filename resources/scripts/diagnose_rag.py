import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.infrastructure.config import load_settings
from src.infrastructure.adapters.faiss.vector_store import FAISSVectorStore

settings = load_settings(ROOT)
vs = FAISSVectorStore(settings)
vs.initialize()

print(f"Coleccion: {settings.faiss_collection}")
print(f"Documentos en la coleccion: {vs.count()}")

docs = vs.search("total de la factura de SIMBA SOFTWARE", top_k=5)
print(f"Documentos recuperados por la busqueda: {len(docs)}")
for d in docs:
    print(f"  - {d.document_id[:12]} score={d.score:.3f} {d.metadata.get('source_file')}")