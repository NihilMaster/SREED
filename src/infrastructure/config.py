from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    root: Path
    input_dir: Path
    processed_dir: Path
    failed_dir: Path
    log_dir: Path
    chroma_dir: Path

    llm_provider: str
    ollama_host: str
    ollama_model: str
    ocr_langs: str
    use_gpu: bool

    gemini_api_key: str
    gemini_model: str


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(project_root: Path | None = None) -> Settings:
    """
    Carga configuracion desde .env.

    Si falta una variable, usa rutas relativas basadas en la raiz del proyecto.
    """

    root_fallback = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    load_dotenv(root_fallback / ".env")

    root = Path(os.getenv("SREED_ROOT", str(root_fallback))).expanduser().resolve()

    return Settings(
        root=root,
        input_dir=Path(os.getenv("SREED_INPUT_DIR", str(root / "input_facturas"))).expanduser(),
        processed_dir=Path(os.getenv("SREED_PROCESSED_DIR", str(root / "facturas_procesadas"))).expanduser(),
        failed_dir=Path(os.getenv("SREED_FAILED_DIR", str(root / "facturas_fallidas"))).expanduser(),
        log_dir=Path(os.getenv("SREED_LOG_DIR", str(root / "logs"))).expanduser(),
        chroma_dir=Path(os.getenv("SREED_CHROMA_DIR", str(root / "data" / "chroma"))).expanduser(),
        llm_provider=os.getenv("SREED_LLM_PROVIDER", "qwen"),
        ollama_host=os.getenv("SREED_OLLAMA_HOST", "http://127.0.0.1:11434"),
        ollama_model=os.getenv("SREED_OLLAMA_MODEL", "qwen2.5:7b"),
        ocr_langs=os.getenv("SREED_OCR_LANGS", "es,en"),
        use_gpu=_as_bool(os.getenv("SREED_USE_GPU", "false"), False),
        gemini_api_key=os.getenv("SREED_GEMINI_API_KEY", ""),
        gemini_model=os.getenv("SREED_GEMINI_MODEL", "gemini-1.5-flash"),
    )