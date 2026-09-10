from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.application.use_cases.process_document import ProcessDocumentUseCase
from src.infrastructure.adapters.local_filesystem_adapter import LocalFileSystemAdapter
from src.infrastructure.adapters.stub_document_processor import StubDocumentProcessor
from src.infrastructure.config import load_settings


st.set_page_config(
    page_title="SREED",
    layout="wide",
)


@st.cache_resource
def get_services():
    settings = load_settings(ROOT)
    file_system = LocalFileSystemAdapter(settings)
    processor = StubDocumentProcessor()
    use_case = ProcessDocumentUseCase(processor, file_system)
    return settings, file_system, use_case


def sanitize_filename(name: str) -> str:
    name = Path(name).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def save_uploaded_file(settings, uploaded_file) -> Path:
    safe_name = sanitize_filename(uploaded_file.name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = settings.input_dir / f"{timestamp}_{safe_name}"
    destination.write_bytes(uploaded_file.getvalue())
    return destination


def render_header(settings) -> None:
    st.title("SREED")
    st.caption("Sistema Resiliente de Extraccion Estructurada de Documentos")
    st.info(
        "Modulo de procesamiento actual: Stub. "
        "El adaptador OpenCV + EasyOCR + Qwen se conectara sobre el mismo puerto."
    )


def render_sidebar(settings) -> None:
    with st.sidebar:
        st.header("Estado")
        st.write(f"Proveedor LLM: {settings.llm_provider}")
        st.write(f"Modelo Ollama: {settings.ollama_model}")
        st.write(f"OCR langs: {settings.ocr_langs}")
        st.write(f"GPU: {settings.use_gpu}")

        st.divider()

        st.write(f"Entrada: {settings.input_dir}")
        st.write(f"Procesados: {settings.processed_dir}")
        st.write(f"Fallidos: {settings.failed_dir}")

        if st.button("Actualizar vista", key="sidebar_refresh"):
            st.rerun()


def render_monitor_tab(settings, file_system) -> None:
    st.subheader("Monitor de carpeta")

    input_files = file_system.list_files(settings.input_dir)
    processed_files = file_system.list_files(settings.processed_dir)
    failed_files = file_system.list_files(settings.failed_dir)

    col1, col2, col3 = st.columns(3)

    col1.metric("Entrada", len(input_files))
    col2.metric("Procesados", len(processed_files))
    col3.metric("Fallidos", len(failed_files))

    if st.button("Actualizar", key="monitor_refresh"):
        st.rerun()

    st.divider()

    st.subheader("Archivos pendientes en entrada")

    if not input_files:
        st.info("No hay archivos pendientes en input_facturas.")
        return

    rows = []

    for path in input_files:
        try:
            stat = path.stat()
            rows.append(
                {
                    "archivo": path.name,
                    "tamano_bytes": stat.st_size,
                    "modificado": datetime.fromtimestamp(stat.st_mtime).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                }
            )
        except OSError:
            continue

    st.dataframe(rows, use_container_width=True)


def render_manual_tab(settings, file_system) -> None:
    st.subheader("Carga manual")

    st.caption(
        "Los archivos se guardan en input_facturas. "
        "El monitor watchdog los detectara y procesara."
    )

    uploaded_files = st.file_uploader(
        "Seleccionar facturas",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        return

    for uploaded_file in uploaded_files:
        st.text(uploaded_file.name)

    if st.button("Guardar en carpeta vigilada"):
        for uploaded_file in uploaded_files:
            destination = save_uploaded_file(settings, uploaded_file)
            st.success(f"Archivo guardado: {destination.name}")

        st.warning("Asegurese de que monitor_rpa.py este activo para procesarlos.")


def render_results_tab(settings, file_system) -> None:
    st.subheader("Resultados")

    option = st.radio(
        "Carpeta",
        ["Procesados", "Fallidos"],
        horizontal=True,
    )

    directory = settings.processed_dir if option == "Procesados" else settings.failed_dir
    files = file_system.list_files(directory)

    if not files:
        st.info("No hay documentos en la carpeta seleccionada.")
        return

    selected = st.selectbox(
        "Documento",
        files,
        format_func=lambda path: path.name,
    )

    if selected is None:
        return

    st.caption(str(selected))

    sidecar = file_system.read_sidecar(selected)

    if sidecar is None:
        st.warning("No existe JSON lateral para este documento.")
        return

    st.json(sidecar)


def render_config_tab(settings) -> None:
    st.subheader("Configuracion")

    gemini_key = "****" if settings.gemini_api_key else ""

    content = f"""
SREED_ROOT={settings.root}
SREED_INPUT_DIR={settings.input_dir}
SREED_PROCESSED_DIR={settings.processed_dir}
SREED_FAILED_DIR={settings.failed_dir}
SREED_LOG_DIR={settings.log_dir}
SREED_CHROMA_DIR={settings.chroma_dir}

SREED_LLM_PROVIDER={settings.llm_provider}
SREED_OLLAMA_HOST={settings.ollama_host}
SREED_OLLAMA_MODEL={settings.ollama_model}
SREED_OCR_LANGS={settings.ocr_langs}
SREED_USE_GPU={settings.use_gpu}

SREED_GEMINI_API_KEY={gemini_key}
SREED_GEMINI_MODEL={settings.gemini_model}
"""

    st.code(content, language="text")
    st.caption("Para modificar estos valores edite el archivo .env y reinicie los procesos.")


def main() -> None:
    settings, file_system, use_case = get_services()

    render_header(settings)
    render_sidebar(settings)

    tab_monitor, tab_manual, tab_results, tab_config = st.tabs(
        [
            "Monitor",
            "Carga manual",
            "Resultados",
            "Configuracion",
        ]
    )

    with tab_monitor:
        render_monitor_tab(settings, file_system)

    with tab_manual:
        render_manual_tab(settings, file_system)

    with tab_results:
        render_results_tab(settings, file_system)

    with tab_config:
        render_config_tab(settings)


main()