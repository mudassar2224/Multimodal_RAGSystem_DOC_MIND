import asyncio
import base64
import csv
import html
import io
import json
import mimetypes
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


# ---------------------------------------------------------------------------
# In-session file preview cache.
#
# The moment a file is selected in the uploader, its raw bytes go here —
# that's what lets video/audio/images/text play or render inline instead of
# just showing a filename. This cache lives only in this browser session
# (Streamlit session_state), so a page refresh or a brand-new session won't
# have it. As a fallback for that case, _doc_bytes_and_mime() also checks
# whether a document row from the database exposes a usable URL (common
# attribute names only — no guessed method calls), and st.video/audio/image
# all accept a URL directly, so that path works too if your object store
# exposes one.
# ---------------------------------------------------------------------------
def _cache_file(thread_id: str, name: str, data: bytes, mime: str | None) -> None:
    if "file_cache" not in st.session_state:
        st.session_state.file_cache = {}
    st.session_state.file_cache.setdefault(thread_id, {})[name] = {"data": data, "mime": mime}


def _doc_bytes_and_mime(doc, thread_id: str):
    label = _doc_label(doc)
    cached = st.session_state.get("file_cache", {}).get(thread_id, {}).get(label)
    if cached:
        return label, cached["mime"], cached["data"], None

    url = None
    for attr in ("url", "download_url", "public_url", "object_url", "presigned_url"):
        value = getattr(doc, attr, None)
        if value:
            url = str(value)
            break
    return label, None, None, url


def _render_single_preview(name: str, mime: str | None, data: bytes | None, url: str | None = None) -> None:
    mime = mime or mimetypes.guess_type(name)[0] or ""
    lower_name = name.lower()

    if data is None and url is None:
        st.info("Preview unavailable in this session — re-upload the file to preview or play it.")
        return

    if mime.startswith("image/"):
        st.image(data if data is not None else url, use_container_width=True)
    elif mime.startswith("video/"):
        st.video(data if data is not None else url)
    elif mime.startswith("audio/"):
        st.audio(data if data is not None else url)
    elif mime == "application/pdf" or lower_name.endswith(".pdf"):
        if data is not None:
            src = f"data:application/pdf;base64,{base64.b64encode(data).decode()}"
        else:
            src = url
        st.markdown(
            f'<iframe src="{src}" width="100%" height="480" style="border:none;"></iframe>',
            unsafe_allow_html=True,
        )
    elif data is not None and (mime in ("text/plain", "text/markdown") or lower_name.endswith((".txt", ".md"))):
        text = data.decode("utf-8", errors="replace")
        st.text(text[:5000])
        if len(text) > 5000:
            st.caption("Preview truncated — showing the first 5,000 characters.")
    elif data is not None and (mime == "application/json" or lower_name.endswith(".json")):
        try:
            st.json(json.loads(data.decode("utf-8")))
        except Exception:  # noqa: BLE001 -- malformed JSON still gets a raw-text fallback
            st.text(data.decode("utf-8", errors="replace")[:5000])
    elif data is not None and (mime == "text/csv" or lower_name.endswith(".csv")):
        try:
            rows = list(csv.reader(io.StringIO(data.decode("utf-8", errors="replace"))))[:25]
            st.table(rows)
            st.caption("Preview limited to the first 25 rows.")
        except Exception:  # noqa: BLE001 -- fall back to plain text if it isn't valid CSV
            st.text(data.decode("utf-8", errors="replace")[:5000])
    elif url:
        st.info(f"No inline preview for this file type ({mime or 'unknown type'}).")
        st.markdown(f"[Open file]({url})")
    else:
        st.info(
            f"No inline preview for this file type ({mime or 'unknown type'}) — "
            "download it to view it locally."
        )

    if data is not None:
        st.download_button(
            "Download",
            data=data,
            file_name=name,
            mime=mime or "application/octet-stream",
            key=f"download-{name}-{len(data)}",
        )


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
if "file_cache" not in st.session_state:
    st.session_state.file_cache = {}

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

# ---------------------------------------------------------------------------
# Playable / viewable preview of every file in this chat — already-processed
# documents plus anything just selected but not yet processed. One tab per
# file so video/audio/PDF/etc. get real width to render in, instead of being
# squeezed into the sidebar.
# ---------------------------------------------------------------------------
existing_names = {_doc_label(doc) for doc in conversation_documents}
preview_entries = []

for doc in conversation_documents:
    label, mime, data, url = _doc_bytes_and_mime(doc, st.session_state.thread_id)
    preview_entries.append({"label": label, "mime": mime, "data": data, "url": url, "pending": False})

if uploads:
    for upload in uploads:
        if upload.name in existing_names:
            continue
        data = upload.getvalue()
        _cache_file(st.session_state.thread_id, upload.name, data, upload.type)
        preview_entries.append(
            {"label": upload.name, "mime": upload.type, "data": data, "url": None, "pending": True}
        )

if preview_entries:
    st.subheader("📎 Files in this chat")
    tab_titles = [
        f"{_file_icon(entry['label'])} {entry['label']}" + (" · pending" if entry["pending"] else "")
        for entry in preview_entries
    ]
    preview_tabs = st.tabs(tab_titles)
    for tab, entry in zip(preview_tabs, preview_entries):
        with tab:
            if entry["pending"]:
                st.caption("Not processed yet — click **Process Files** above to index it.")
            _render_single_preview(entry["label"], entry["mime"], entry["data"], entry["url"])

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
