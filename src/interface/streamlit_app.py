from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.application.use_cases.answer_question import AnswerQuestionUseCase
from src.domain.models import ChatMessage, ChatRole
from src.infrastructure.adapters.chroma import ChromaVectorStore
from src.infrastructure.adapters.filesystem import LocalFileSystemAdapter
from src.infrastructure.adapters.llm import QwenOllamaProvider
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

    llm_timeout = getattr(settings, "llm_timeout", 1800)

    llm_provider = QwenOllamaProvider(
        primary_model=settings.primary_model,
        fallback_model=settings.fallback_model,
        host=settings.ollama_host,
        context_window=settings.context_window,
        timeout=llm_timeout,
    )

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
    st.caption("Sistema Resiliente de Extracción Estructurada de Documentos")
    st.info(
        "Módulo de procesamiento actual: OCR Real + Estrategia Híbrida Qwen/Gemini. "
        "El sistema preprocesa la imagen, extrae texto y estructura los datos semánticamente."
    )


def render_sidebar(settings, vector_store) -> None:
    with st.sidebar:
        st.header("Estado del Sistema")

        st.write(f"Modelo principal: {settings.primary_model}")
        st.write(f"Modelo fallback: {settings.fallback_model}")
        st.write(f"Modo LLM: {settings.llm_mode}")
        st.write(f"Proveedor RAG: qwen")
        st.write(f"Ollama Host: {settings.ollama_host}")
        st.write(f"Contexto: {settings.context_window} tokens")
        st.write(f"Timeout LLM: {getattr(settings, 'llm_timeout', 1800)}s")

        st.divider()

        st.write(f"OCR idiomas: {', '.join(settings.ocr_languages)}")
        st.write(f"GPU OCR: {'Sí' if settings.use_gpu else 'No'}")
        st.write(f"PDF DPI: {settings.pdf_dpi}")

        st.divider()

        try:
            document_count = vector_store.count()
            st.write(f"Documentos indexados: {document_count}")
        except Exception:
            st.write("Documentos indexados: error")

        st.write(f"Colección: {settings.chroma_collection}")

        st.divider()

        st.write(f"Entrada: {settings.input_dir}")
        st.write(f"Procesados: {settings.processed_dir}")
        st.write(f"Fallidos: {settings.failed_dir}")

        if st.button("Actualizar vista", key="sidebar_refresh"):
            st.rerun()

def filter_gitkeep(files):
    return [f for f in files if f.name != ".gitkeep"]

def render_monitor_tab(settings, file_system) -> None:
    st.subheader("Monitor de carpeta")

    input_files = filter_gitkeep(file_system.list_files(settings.input_dir))
    processed_files = filter_gitkeep(file_system.list_files(settings.processed_dir))
    failed_files = filter_gitkeep(file_system.list_files(settings.failed_dir))

    col1, col2, col3 = st.columns(3)
    col1.metric("Entrada", len(input_files))
    col2.metric("Procesados", len(processed_files))
    col3.metric("Fallidos", len(failed_files))

    if st.button("Actualizar", key="monitor_refresh"):
        st.rerun()

    st.divider()
    st.subheader("Archivos pendientes en entrada")

    if not input_files:
        st.info("No hay archivos pendientes en la carpeta de entrada.")
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
    st.subheader("Carga manual con seguimiento en tiempo real")
    st.caption(
        "Los archivos se procesarán sincrónicamente con monitoreo activo para reflejar "
        "el progreso de cada etapa."
    )

    uploaded_files = st.file_uploader(
        "Seleccionar facturas",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        return

    if st.button("Iniciar procesamiento"):
        for uploaded_file in uploaded_files:
            with st.status(f"Procesando: {uploaded_file.name}", expanded=True) as status:
                status.update(label=f"1. Guardando {uploaded_file.name} en carpeta de entrada...")
                destination = save_uploaded_file(settings, uploaded_file)
                time.sleep(0.5)

                status.update(label="2. Esperando detección del monitor y ejecución de OCR...")
                processed_path = None
                failed_path = None
                max_wait = 300  # Máximo 5 minutos de espera por archivo

                # Extracción del stem sin extensión
                uploaded_stem = Path(uploaded_file.name).stem

                for _ in range(max_wait):
                    time.sleep(1)
                    proc_files = [
                        f for f in settings.processed_dir.iterdir()
                        if f.name.endswith(uploaded_file.name)
                        or (f.stem.endswith(uploaded_stem) and f.suffix in ['.json', '.jpg', '.png', '.jpeg', '.pdf'])
                    ]
                    fail_files = [
                        f for f in settings.failed_dir.iterdir()
                        if f.name.endswith(uploaded_file.name)
                        or (f.stem.endswith(uploaded_stem) and f.suffix in ['.json', '.jpg', '.png', '.jpeg', '.pdf'])
                    ]

                    if proc_files:
                        processed_path = proc_files[0]
                        break
                    if fail_files:
                        failed_path = fail_files[0]
                        break

                if processed_path:
                    status.update(
                        label="3. Extracción y estructuración LLM completada exitosamente.",
                        state="complete",
                    )
                    st.success(f"Archivo procesado: {processed_path.name}")
                elif failed_path:
                    status.update(
                        label="3. El procesamiento falló. Revisa la carpeta de fallidos.",
                        state="error",
                    )
                    st.error(f"Archivo fallido: {failed_path.name}")
                else:
                    status.update(
                        label="3. Tiempo de espera agotado. El proceso continúa en segundo plano.",
                        state="running",
                    )
                    st.warning("El archivo está en cola. Revisa la pestaña 'Resultados' en breve.")


def render_results_tab(settings, file_system) -> None:
    st.subheader("Resultados")

    option = st.radio(
        "Carpeta",
        ["Procesados", "Fallidos"],
        horizontal=True,
    )

    directory = settings.processed_dir if option == "Procesados" else settings.failed_dir
    files = file_system.list_files(directory)

    # Excluir .gitkeep
    files = [f for f in files if f.name != ".gitkeep"]

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

    st.caption(f"Ruta: {selected}")
    sidecar = file_system.read_sidecar(selected)

    if sidecar is None:
        st.warning("No existe JSON lateral para este documento.")
        return

    st.subheader("Datos Estructurados (JSON)")
    if isinstance(sidecar, (dict, list)):
        formatted_json = json.dumps(sidecar, indent=2, ensure_ascii=False)
    else:
        formatted_json = str(sidecar)

    st.code(formatted_json, language="json")


def render_rag_tab(settings, vector_store, answer_use_case) -> None:
    st.subheader("Asistente analítico (RAG)")
    st.caption(
        "Proveedor RAG actual: Qwen local. "
        "El sistema recupera documentos de la base vectorial y genera respuestas basadas en el contexto."
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
        with st.expander("Detalle de recuperación (Última consulta)"):
            rag_answer = st.session_state.rag_last_answer
            st.write(f"Proveedor: {getattr(rag_answer, 'provider', 'Qwen')}")
            st.write(f"Pregunta: {rag_answer.question}")

            if rag_answer.retrieved_documents:
                rows = []
                for doc in rag_answer.retrieved_documents:
                    rows.append(
                        {
                            "document_id": doc.document_id,
                            "source_file": doc.source_file,
                            "score": getattr(doc, 'score', 'N/A'),
                            "snippet": doc.snippet[:200],
                        }
                    )
                st.dataframe(rows, use_container_width=True)

                for doc in rag_answer.retrieved_documents:
                    st.markdown(f"**Documento:** {doc.source_file or doc.document_id}")
                    st.code(doc.snippet, language="text")
            else:
                st.info("No se recuperaron documentos.")

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Limpiar chat"):
            st.session_state.rag_chat = []
            st.session_state.rag_last_answer = None
            st.rerun()

    question = st.chat_input("Escribe una pregunta sobre las facturas procesadas")

    if question:
        st.session_state.rag_chat.append(ChatMessage(role=ChatRole.USER, content=question))

        with st.status("Procesando consulta RAG...", expanded=True) as status:
            status.update(label="Buscando documentos relevantes en la base vectorial...")
            time.sleep(0.3)
            status.update(label="Generando respuesta con el modelo LLM...")

            try:
                rag_answer = answer_use_case.execute(question)
                status.update(label="Respuesta generada exitosamente.", state="complete")
            except Exception as exc:
                status.update(label=f"Error consultando RAG: {exc}", state="error")
                rag_answer = None

        if rag_answer:
            assistant_text = rag_answer.answer

            if rag_answer.retrieved_documents:
                sources = ", ".join(
                    doc.source_file or doc.document_id
                    for doc in rag_answer.retrieved_documents[:5]
                )
                assistant_text += f"\n\n**Fuentes recuperadas:** {sources}"

            st.session_state.rag_chat.append(
                ChatMessage(role=ChatRole.ASSISTANT, content=assistant_text)
            )
            st.session_state.rag_last_answer = rag_answer

        st.rerun()


def render_config_tab(settings) -> None:
    st.subheader("Configuración")

    gemini_key = "****" if settings.gemini_api_key else "(no configurada)"

    content = f"""
SREED_ROOT={settings.root}
SREED_INPUT_DIR={settings.input_dir}
SREED_PROCESSED_DIR={settings.processed_dir}
SREED_FAILED_DIR={settings.failed_dir}
SREED_LOG_DIR={settings.log_dir}
SREED_CHROMA_DIR={settings.chroma_dir}
SREED_MODELS_DIR={settings.models_dir}

SREED_OLLAMA_HOST={settings.ollama_host}
SREED_PRIMARY_MODEL={settings.primary_model}
SREED_FALLBACK_MODEL={settings.fallback_model}
SREED_CONTEXT_WINDOW={settings.context_window}
SREED_LLM_TIMEOUT={getattr(settings, 'llm_timeout', 1800)}

SREED_OCR_LANGUAGES={','.join(settings.ocr_languages)}
SREED_USE_GPU={settings.use_gpu}
SREED_PDF_DPI={settings.pdf_dpi}

SREED_CHROMA_COLLECTION={settings.chroma_collection}
SREED_RAG_TOP_K={settings.rag_top_k}

SREED_GEMINI_API_KEY={gemini_key}
SREED_GEMINI_MODEL={settings.gemini_model}

SREED_LLM_MODE={settings.llm_mode}
SREED_LOG_LEVEL={settings.log_level}
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
            "Configuración",
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


if __name__ == "__main__":
    main()