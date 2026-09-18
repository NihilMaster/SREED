from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


VALID_LLM_MODES = {
    "qwen_only",
    "qwen_gemini",
}


@dataclass(frozen=True)
class Settings:
    # Rutas de negocio
    root: Path
    input_dir: Path
    processed_dir: Path
    failed_dir: Path
    
    # Rutas técnicas
    chroma_dir: Path
    log_dir: Path
    models_dir: Path
    
    # Ollama / Qwen
    ollama_host: str
    primary_model: str
    fallback_model: str
    context_window: int
    llm_timeout: int
    
    # OCR
    ocr_languages: list
    use_gpu: bool
    pdf_dpi: int
    
    # ChromaDB / RAG
    chroma_collection: str
    rag_top_k: int
    
    # Gemini (opcional)
    gemini_api_key: str
    gemini_model: str
    
    # Modo de operación
    llm_mode: str
    
    # Logging
    log_level: str


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int = 0) -> int:
    if value is None:
        return default

    try:
        return int(value.strip())
    except ValueError:
        return default


def _as_list(value: str | None, default: list = None) -> list:
    if value is None:
        return default or []
    return [item.strip() for item in value.split(",") if item.strip()]


def load_settings(project_root: Path | None = None) -> Settings:
    """
    Carga configuración desde .env.

    Si falta una variable, usa rutas relativas basadas en la raíz del proyecto.
    """

    root_fallback = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    load_dotenv(root_fallback / ".env")

    root = Path(os.getenv("SREED_ROOT", str(root_fallback))).expanduser().resolve()

    settings = Settings(
        # Rutas de negocio
        root=root,
        input_dir=Path(os.getenv("SREED_INPUT_DIR", str(root / "invoices" / "input"))).expanduser(),
        processed_dir=Path(os.getenv("SREED_PROCESSED_DIR", str(root / "invoices" / "processed"))).expanduser(),
        failed_dir=Path(os.getenv("SREED_FAILED_DIR", str(root / "invoices" / "failed"))).expanduser(),
        
        # Rutas técnicas
        chroma_dir=Path(os.getenv("SREED_CHROMA_DIR", str(root / "resources" / "chroma"))).expanduser(),
        log_dir=Path(os.getenv("SREED_LOG_DIR", str(root / "resources" / "logs"))).expanduser(),
        models_dir=Path(os.getenv("SREED_MODELS_DIR", str(root / "resources" / "models"))).expanduser(),
        
        # Ollama / Qwen
        ollama_host=os.getenv("SREED_OLLAMA_HOST", "http://localhost:11434"),
        primary_model=os.getenv("SREED_PRIMARY_MODEL", "qwen3.5:9b"),
        fallback_model=os.getenv("SREED_FALLBACK_MODEL", "qwen2.5:7b"),
        context_window=_as_int(os.getenv("SREED_CONTEXT_WINDOW", "4096"), 4096),
        llm_timeout=_as_int(os.getenv("SREED_LLM_TIMEOUT", "120"), 120),
        
        # OCR
        ocr_languages=_as_list(os.getenv("SREED_OCR_LANGUAGES", "es,en"), ["es", "en"]),
        use_gpu=_as_bool(os.getenv("SREED_USE_GPU", "true"), True),
        pdf_dpi=_as_int(os.getenv("SREED_PDF_DPI", "200"), 200),
        
        # ChromaDB / RAG
        chroma_collection=os.getenv("SREED_CHROMA_COLLECTION", "sreed_documents"),
        rag_top_k=_as_int(os.getenv("SREED_RAG_TOP_K", "5"), 5),
        
        # Gemini (opcional)
        gemini_api_key=os.getenv("SREED_GEMINI_API_KEY", ""),
        gemini_model=os.getenv("SREED_GEMINI_MODEL", "gemini-3.6-flash"),
        
        # Modo de operación
        llm_mode=os.getenv("SREED_LLM_MODE", "qwen_only"),
        
        # Logging
        log_level=os.getenv("SREED_LOG_LEVEL", "INFO"),
    )

    # Crear directorios si no existen
    directories_to_create = [
        settings.input_dir,
        settings.processed_dir,
        settings.failed_dir,
        settings.chroma_dir,
        settings.log_dir,
        settings.models_dir,
    ]
    
    for directory in directories_to_create:
        directory.mkdir(parents=True, exist_ok=True)

    _validate_settings(settings)

    return settings


def _validate_settings(settings: Settings) -> None:
    """
    Validaciones de configuración.

    Regla importante:
    - Gemini solo no está permitido.
    - Solo se permiten modos donde Qwen sea principal.
    """

    if settings.llm_mode not in VALID_LLM_MODES:
        raise ValueError(
            "SREED_LLM_MODE inválido. "
            "Valores permitidos: qwen_only, qwen_gemini. "
            "Gemini solo no está permitido."
        )

    # Si se usa modo qwen_gemini, debe haber API key de Gemini
    if settings.llm_mode == "qwen_gemini" and not settings.gemini_api_key:
        raise ValueError(
            "SREED_GEMINI_API_KEY es requerida cuando SREED_LLM_MODE=qwen_gemini"
        )