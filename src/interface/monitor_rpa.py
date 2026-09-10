from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.application.use_cases.index_document import IndexDocumentUseCase
from src.application.use_cases.process_and_index_document import (
    ProcessAndIndexDocumentUseCase,
)
from src.application.use_cases.process_document import ProcessDocumentUseCase
from src.infrastructure.adapters.chroma_vector_store import ChromaVectorStore
from src.infrastructure.adapters.local_filesystem_adapter import LocalFileSystemAdapter
from src.infrastructure.adapters.stub_document_processor import StubDocumentProcessor
from src.infrastructure.adapters.watchdog_adapter import WatchdogFolderMonitor
from src.infrastructure.config import load_settings


def configure_logging(settings) -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    log_file = settings.log_dir / "monitor_rpa.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def main() -> None:
    settings = load_settings(ROOT)
    configure_logging(settings)

    logger = logging.getLogger("sreed.monitor")

    file_system = LocalFileSystemAdapter(settings)

    processor = StubDocumentProcessor()

    process_use_case = ProcessDocumentUseCase(
        processor=processor,
        file_system=file_system,
    )

    vector_store = ChromaVectorStore(settings)

    try:
        vector_store.initialize()
        logger.info("ChromaDB inicializado correctamente desde monitor_rpa.")
    except Exception:
        logger.exception(
            "No se pudo inicializar ChromaDB. "
            "El monitor seguira activo, pero la indexacion puede fallar."
        )

    index_use_case = IndexDocumentUseCase(vector_store)

    use_case = ProcessAndIndexDocumentUseCase(
        process_document=process_use_case,
        index_document=index_use_case,
    )

    monitor = WatchdogFolderMonitor(
        settings=settings,
        use_case=use_case,
        file_system=file_system,
    )

    monitor.start()

    logger.info("SREED monitor activo. Presione Ctrl+C para detener.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Deteniendo monitor por accion del usuario.")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()