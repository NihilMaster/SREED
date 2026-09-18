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
from src.infrastructure.adapters.faiss.vector_store import FAISSVectorStore
from src.infrastructure.adapters.filesystem import LocalFileSystemAdapter
from src.infrastructure.adapters.llm import GeminiVisionProvider, QwenOllamaProvider
from src.infrastructure.adapters.ocr import OCRRealDocumentProcessor
from src.infrastructure.adapters.rpa import WatchdogAdapter
from src.infrastructure.config import load_settings


def configure_logging(settings) -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    log_file = settings.log_dir / "monitor_rpa.log"
    
    # Configuración base
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    
    # Logger específico para RAG (más verboso)
    rag_logger = logging.getLogger("src.infrastructure.adapters.llm.qwen_ollama_provider")
    rag_logger.setLevel(logging.DEBUG)


def main() -> None:
    settings = load_settings(ROOT)
    configure_logging(settings)

    logger = logging.getLogger("sreed.monitor")

    file_system = LocalFileSystemAdapter(settings)

    # Inicializar OCR real
    logger.info("Inicializando OCR real...")
    ocr_processor = OCRRealDocumentProcessor(
        languages=["es", "en"],
        use_gpu=True
    )

    # Inicializar Qwen
    logger.info("Inicializando Qwen Ollama Provider...")
    qwen_provider = QwenOllamaProvider(
        # primary_model="qwen3.5:9b-4k-OCR:latest",  
        primary_model="qwen3.5:9b",
        fallback_model="qwen2.5:7b",
        host=settings.ollama_host,
        context_window=4096,
        timeout=900,
    )

    # Determinar qué proveedor LLM usar según configuración
    if settings.llm_mode == "qwen_gemini" and settings.gemini_api_key:
        logger.info("Modo híbrido Qwen+Gemini activado")

        # Inicializar Gemini Vision
        gemini_provider = GeminiVisionProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        )

        # Usar estrategia híbrida
        from src.infrastructure.adapters.llm.qwen_gemini_strategy import QwenGeminiStrategy
        llm_provider = QwenGeminiStrategy(
            qwen_provider=qwen_provider,
            gemini_provider=gemini_provider,
        )
    else:
        logger.info("Modo Qwen-only activado")
        llm_provider = qwen_provider

    # Caso de uso de procesamiento con OCR + LLM
    process_use_case = ProcessDocumentUseCase(
        processor=ocr_processor,
        llm_provider=llm_provider,
        file_system=file_system,
    )

    # Vector store para FAISS
    vector_store = FAISSVectorStore(settings)

    try:
        vector_store.initialize()
        logger.info("FAISS Vector Store inicializado correctamente desde monitor_rpa.")
    except Exception:
        logger.exception(
            "No se pudo inicializar FAISS Vector Store. "
            "El monitor seguirá activo, pero la indexación puede fallar."
        )

    index_use_case = IndexDocumentUseCase(vector_store)

    # Caso de uso compuesto: procesar + indexar
    use_case = ProcessAndIndexDocumentUseCase(
        process_document=process_use_case,
        index_document=index_use_case,
    )

    # Monitor con watchdog
    monitor = WatchdogAdapter(
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
        logger.info("Deteniendo monitor por acción del usuario.")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()