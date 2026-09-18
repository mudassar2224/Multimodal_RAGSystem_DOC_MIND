import asyncio
import html
import json
import os
import uuid

import streamlit as st

from docmind.core.config import get_settings
from docmind.graph.workflow import DocMindGraph
from docmind.ingestion.service import IngestionService
from docmind.memory.service import MemoryService
from docmind.models.hf import (
    ColQwenClient,
    HuggingFaceASRClient,
    HuggingFaceEmbeddingClient,
    QwenClient,
)
from docmind.retrieval.hybrid import HybridRetriever
from docmind.storage.database import Database
from docmind.storage.object_store import get_object_store

# ---------------------------------------------------------------------------
# st.set_page_config MUST be the very first Streamlit command that runs.
# It was previously called after a function definition and a helper call —
# harmless today, but moving it here removes any risk if that helper (or
# anything above it) ever grows a Streamlit call of its own.
# ---------------------------------------------------------------------------
st.set_page_config(page_title="DocMind", page_icon="🧠", layout="wide")


def load_streamlit_secrets() -> None:
    """Expose Community Cloud secrets to the shared Pydantic settings loader."""
    for name in (
        "HF_TOKEN",
        "HF_PROVIDER",
        "QWEN_MODEL",
        "TEXT_EMBEDDING_MODEL",
        "VISUAL_RETRIEVAL_MODEL",
        "VISUAL_ENDPOINT_URL",
        "ASR_MODEL",
        "DATABASE_URL",
        "OBJECT_STORAGE_BACKEND",
        "LOCAL_OBJECT_DIR",
        "S3_BUCKET",
        "S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "MAX_UPLOAD_MB",
        "VIDEO_FRAME_INTERVAL_SECONDS",
    ):
        try:
            if name in st.secrets:
                os.environ.setdefault(name, str(st.secrets[name]))
        except Exception:
            # No secrets.toml locally and nothing injected by Community Cloud —
            # fall back to whatever is already in the process environment
            # instead of crashing the whole app on import.
            break


load_streamlit_secrets()


@st.cache_resource(show_spinner=False)
def services() -> tuple[Database, IngestionService, DocMindGraph]:
    settings = get_settings()
    database = Database(settings)
    database.create_all()
    objects = get_object_store(settings)
    embeddings = HuggingFaceEmbeddingClient(settings)
    ingestion = IngestionService(
        settings,
        database,
        objects,
        embeddings,
        ColQwenClient(settings),
        HuggingFaceASRClient(settings),
    )
    graph = DocMindGraph(
        database,
        MemoryService(database),
        HybridRetriever(database, embeddings),
        QwenClient(settings),
        objects,
    )
    return database, ingestion, graph


# ---------------------------------------------------------------------------
# Front-end helpers only. These deliberately use best-effort attribute
# lookups (getattr / hasattr) rather than assuming exact field names on
# Database.documents() rows or the sources_json schema, since that schema
# lives in the docmind package and wasn't part of what was shared here.
# If any of these guesses are off, the app will still render — just with a
# plainer label — instead of crashing.
# ---------------------------------------------------------------------------
_FILE_ICONS = {
    "pdf": "📄", "docx": "📝", "doc": "📝", "pptx": "📊", "ppt": "📊",
    "xlsx": "📈", "xls": "📈", "txt": "📄", "md": "📄", "csv": "📈",
    "json": "🧾", "jpg": "🖼️", "jpeg": "🖼️", "png": "🖼️", "webp": "🖼️",
    "mp3": "🎵", "wav": "🎵", "m4a": "🎵",
    "mp4": "🎬", "mov": "🎬", "mkv": "🎬", "webm": "🎬",
}


def _doc_label(doc) -> str:
    for attr in ("filename", "name", "title", "display_name"):
        value = getattr(doc, attr, None)
        if value:
            return str(value)
    if isinstance(doc, dict):
        return str(doc.get("filename") or doc.get("name") or doc)
    return str(doc)


def _file_icon(label: str) -> str:
    ext = label.rsplit(".", 1)[-1].lower() if "." in label else ""
    return _FILE_ICONS.get(ext, "📁")


def _parse_source_labels(sources_json: str) -> list[str]:
    """Turn a stored sources_json string into a flat list of display
    strings, tolerating whatever shape it was actually saved in."""
    if not sources_json or sources_json == "[]":
        return []
    try:
        data = json.loads(sources_json)
    except (json.JSONDecodeError, TypeError):
        return [sources_json]
    labels: list[str] = []
    for item in data:
        if isinstance(item, str):
            labels.append(item)
        elif isinstance(item, dict):
            labels.append(str(item.get("display") or item.get("label") or item))
        else:
            labels.append(str(item))
    return labels


def _render_source_chips(labels: list[str]) -> None:
    if not labels:
        return
    chips = "".join(f'<span class="dm-chip">{html.escape(label)}</span>' for label in labels)
    st.markdown(f'<div class="dm-chip-row">{chips}</div>', unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .dm-chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 2px 0; }
    .dm-chip {
        background: rgba(120, 120, 255, 0.12);
        border: 1px solid rgba(120, 120, 255, 0.35);
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.78rem;
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 DocMind")
st.caption("Persistent, source-grounded multimodal RAG — hosted inference only")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

try:
    database, ingestion, graph = services()
except Exception as error:  # noqa: BLE001 -- show deployment configuration failures in the UI
    st.error(f"Configuration error: {error}")
    st.stop()

with st.sidebar:
    st.subheader("This conversation")
    try:
        conversation_documents = database.documents(st.session_state.thread_id)
    except Exception as error:  # noqa: BLE001
        conversation_documents = []
        st.error(f"Couldn't load this conversation's files: {error}")

    st.caption(f"{len(conversation_documents)} file(s) available to this chat")
    for doc in conversation_documents:
        label = _doc_label(doc)
        st.markdown(f"{_file_icon(label)} {label}")

    if st.button("New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.uploader_key += 1
        st.session_state.last_sources = []
        st.rerun()

with st.expander("＋ Add files to this chat", expanded=False):
    uploads = st.file_uploader(
        "Upload documents, images, audio, or video",
        accept_multiple_files=True,
        type=[
            "pdf", "docx", "pptx", "xlsx", "txt", "md", "csv", "json",
            "jpg", "jpeg", "png", "webp",
            "mp3", "wav", "m4a",
            "mp4", "mov", "mkv", "webm",
        ],
        key=f"uploads-{st.session_state.thread_id}-{st.session_state.uploader_key}",
    )

    col_process, col_cancel = st.columns(2)
    process_clicked = col_process.button(
        "Process Files", type="primary", use_container_width=True, disabled=not uploads
    )
    cancel_clicked = col_cancel.button(
        "Cancel", use_container_width=True, disabled=not uploads
    )

    if cancel_clicked:
        # st.file_uploader has no imperative "clear" call — bumping the key
        # mounts a fresh widget instance, which is what actually empties it.
        st.session_state.uploader_key += 1
        st.rerun()

    if process_clicked and uploads:
        progress = st.progress(0)
        successful_uploads = 0
        for index, upload in enumerate(uploads, start=1):
            try:
                with st.spinner(f"Indexing {upload.name}…"):
                    asyncio.run(
                        ingestion.index_upload(
                            st.session_state.thread_id,
                            upload.name,
                            upload.getvalue(),
                            upload.type,
                        )
                    )
                successful_uploads += 1
            except Exception as error:  # noqa: BLE001 -- each upload must fail independently
                st.error(f"{upload.name}: {error}")
            progress.progress(index / len(uploads))

        if successful_uploads:
            st.success(f"Added {successful_uploads} file(s) to this chat.")
            st.session_state.uploader_key += 1
            st.rerun()
        if successful_uploads != len(uploads):
            st.warning("Failed files were not indexed. Fix the displayed issue, then upload them again.")

chat_col, evidence_col = st.columns([3, 1])

with chat_col:
    try:
        history = database.messages(st.session_state.thread_id, limit=100)
    except Exception as error:  # noqa: BLE001
        history = []
        st.error(f"Couldn't load chat history: {error}")

    for message in history:
        with st.chat_message(message.role):
            st.write(message.content)
            _render_source_chips(_parse_source_labels(getattr(message, "sources_json", "[]")))

    if question := st.chat_input("Ask about your indexed files"):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            result = None
            with st.spinner("Retrieving evidence and generating an answer…"):
                try:
                    result = asyncio.run(graph.ask(st.session_state.thread_id, question))
                except Exception as error:  # noqa: BLE001 -- hosted services return provider-specific exceptions
                    st.error(f"Answer failed: {error}")

            if result is not None:
                st.write(result.get("answer", ""))
                source_labels: list[str] = []
                for source in result.get("sources", []):
                    try:
                        source_labels.append(source.display())
                    except Exception:  # noqa: BLE001 -- a bad source object should never break the answer
                        source_labels.append(str(source))
                _render_source_chips(source_labels)
                st.session_state.last_sources = source_labels

with evidence_col:
    st.subheader("Context & Sources")
    if st.session_state.last_sources:
        for label in st.session_state.last_sources:
            st.markdown(f"- {label}")
    else:
        st.caption("Ask a question to see the evidence behind the answer here.")
