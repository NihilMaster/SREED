import sys
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.infrastructure.config import load_settings
from src.infrastructure.adapters.faiss.vector_store import FAISSVectorStore
from src.application.use_cases.index_document import IndexDocumentUseCase
from src.domain.models import DocumentResult, DocumentStatus


def main() -> None:
    settings = load_settings(ROOT)

    vector_store = FAISSVectorStore(settings)

    if "--reset" in sys.argv:
        print("Forzando reset completo de FAISS Vector Store...")
        vector_store.reset()
    else:
        vector_store.initialize()

    index_use_case = IndexDocumentUseCase(vector_store)

    sidecars = sorted(settings.processed_dir.glob("*.json"))
    print(f"Reindexando {len(sidecars)} documentos...")

    indexed = 0
    skipped = 0
    legacy = 0

    for sidecar_path in sidecars:
        # 1) Leer JSON
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            skipped += 1
            print(f"✗ {sidecar_path.name} (JSON inválido: {exc})")
            continue

        # 2) Validar campos mínimos (los ejemplos legacy no tienen document_id)
        document_id = data.get("document_id")
        if not document_id:
            legacy += 1
            print(f"⚠ {sidecar_path.name} (sin 'document_id' — legacy, omitido)")
            continue

        # 3) Reconstruir DocumentResult tolerando campos opcionales
        try:
            created_raw = data.get("created_at")
            created_at = (
                datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
                if created_raw
                else datetime.now(timezone.utc)
            )
            source_raw = data.get("source_path")
            final_raw = data.get("final_path")

            result = DocumentResult(
                document_id=document_id,
                source_path=Path(source_raw) if source_raw else sidecar_path,
                final_path=Path(final_raw) if final_raw else sidecar_path,
                status=DocumentStatus.SUCCESS,
                message=data.get("message", ""),
                payload=data.get("payload", {}),
                created_at=created_at,
            )

            if index_use_case.execute(result):
                indexed += 1
                print(f"✓ {sidecar_path.name}")
            else:
                skipped += 1
                print(f"✗ {sidecar_path.name} (falló indexación)")
        except Exception as exc:
            skipped += 1
            print(f"✗ {sidecar_path.name} (error: {exc})")

    print(
        f"\nReindexación completada: {indexed} indexados, "
        f"{skipped} omitidos, {legacy} legacy inválidos"
    )
    print(f"Total en FAISS Vector Store: {vector_store.count()}")


if __name__ == "__main__":
    main()