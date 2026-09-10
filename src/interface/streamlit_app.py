from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.application.use_cases.answer_question import AnswerQuestionUseCase
from src.domain.models import ChatMessage, ChatRole
from src.infrastructure.adapters.chroma_vector_store import ChromaVectorStore
from src.infrastructure.adapters.local_filesystem_adapter import LocalFileSystemAdapter
from src.infrastructure.adapters.mock_llm_provider import MockLLMProvider
from src.infrastructure.config import load_settings


st.set_page_config(
    page_title="SREED",
    layout="wide",
)


@st.cache_resource
def get_services():
    settings = load_settings(ROOT)

    file_system = LocalFileSystemAdapter(settings)

    vector_store = ChromaVectorStore(settings)
    vector_store.initialize()

    llm_provider = MockLLMProvider()

    answer_use_case = AnswerQuestionUseCase(
        vector_store=vector_store,
        llm_provider=llm_provider,
        top_k=settings.rag_top_k,
    )

    return settings, file_system, vector_store, answer_use_case


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


def render_sidebar(settings, vector_store) -> None:
    with st.sidebar:
        st.header("Estado")

        st.write(f"Proveedor LLM: {settings.llm_provider}")
        st.write(f"Modo LLM: {settings.llm_mode}")
        st.write(f"Proveedor RAG: {settings.rag_provider}")
        st.write(f"Modelo Ollama: {settings.ollama_model}")
        st.write(f"OCR langs: {settings.ocr_langs}")
        st.write(f"GPU: {settings.use_gpu}")

        st.divider()

        try:
            document_count = vector_store.count()
            st.write(f"Documentos indexados: {document_count}")
        except Exception:
            st.write("Documentos indexados: error")

        st.write(f"Coleccion: {settings.chroma_collection}")

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


def render_rag_tab(settings, vector_store, answer_use_case) -> None:
    st.subheader("Asistente analítico (RAG)")

    st.caption(
        "Proveedor RAG actual: mock. "
        "Cuando exista Qwen local, este proveedor se reemplazara por QwenLLMProvider. "
        "En modo hibrido, se usara una estrategia Qwen + Gemini, nunca Gemini solo."
    )

    try:
        document_count = vector_store.count()
        st.write(f"Documentos indexados: {document_count}")
    except Exception:
        st.error("No fue posible consultar la cantidad de documentos indexados.")

    if "rag_chat" not in st.session_state:
        st.session_state.rag_chat = []

    if "rag_last_answer" not in st.session_state:
        st.session_state.rag_last_answer = None

    for message in st.session_state.rag_chat:
        with st.chat_message(message.role.value):
            st.write(message.content)

    if st.session_state.rag_last_answer is not None:
        with st.expander("Detalle de recuperacion"):
            rag_answer = st.session_state.rag_last_answer

            st.write(f"Proveedor: {rag_answer.provider}")
            st.write(f"Pregunta: {rag_answer.question}")

            if rag_answer.retrieved_documents:
                rows = []

                for doc in rag_answer.retrieved_documents:
                    rows.append(
                        {
                            "document_id": doc.document_id,
                            "source_file": doc.source_file,
                            "score": doc.score,
                            "snippet": doc.snippet[:200],
                        }
                    )

                st.dataframe(rows, use_container_width=True)

                for doc in rag_answer.retrieved_documents:
                    st.markdown(f"Documento: {doc.source_file or doc.document_id}")
                    st.code(doc.snippet, language="text")
            else:
                st.info("No se recuperaron documentos.")

    col_clear, _ = st.columns([1, 5])

    with col_clear:
        if st.button("Limpiar chat"):
            st.session_state.rag_chat = []
            st.session_state.rag_last_answer = None
            st.rerun()

    question = st.chat_input(
        "Escribe una pregunta sobre las facturas procesadas"
    )

    if question:
        user_message = ChatMessage(
            role=ChatRole.USER,
            content=question,
        )

        st.session_state.rag_chat.append(user_message)

        try:
            with st.spinner("Consultando documentos..."):
                rag_answer = answer_use_case.execute(question)

            assistant_text = rag_answer.answer

            if rag_answer.retrieved_documents:
                sources = ", ".join(
                    doc.source_file or doc.document_id
                    for doc in rag_answer.retrieved_documents[:5]
                )
                assistant_text += f"\n\nFuentes recuperadas: {sources}"

            assistant_message = ChatMessage(
                role=ChatRole.ASSISTANT,
                content=assistant_text,
            )

            st.session_state.rag_chat.append(assistant_message)
            st.session_state.rag_last_answer = rag_answer

        except Exception as exc:
            error_message = ChatMessage(
                role=ChatRole.ASSISTANT,
                content=f"Error consultando RAG: {exc}",
            )
            st.session_state.rag_chat.append(error_message)
            st.session_state.rag_last_answer = None

        st.rerun()


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
SREED_LLM_MODE={settings.llm_mode}
SREED_RAG_PROVIDER={settings.rag_provider}
SREED_RAG_TOP_K={settings.rag_top_k}
SREED_CHROMA_COLLECTION={settings.chroma_collection}

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
    settings, file_system, vector_store, answer_use_case = get_services()

    render_header(settings)
    render_sidebar(settings, vector_store)

    tab_monitor, tab_manual, tab_results, tab_rag, tab_config = st.tabs(
        [
            "Monitor",
            "Carga manual",
            "Resultados",
            "RAG",
            "Configuracion",
        ]
    )

    with tab_monitor:
        render_monitor_tab(settings, file_system)

    with tab_manual:
        render_manual_tab(settings, file_system)

    with tab_results:
        render_results_tab(settings, file_system)

    with tab_rag:
        render_rag_tab(settings, vector_store, answer_use_case)

    with tab_config:
        render_config_tab(settings)


main()