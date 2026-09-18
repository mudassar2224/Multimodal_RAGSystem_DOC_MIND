import asyncio
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


@st.cache_resource
def services() -> tuple[Database, IngestionService, DocMindGraph]:
    settings = get_settings()
    database = Database(settings)
    database.create_all()
    objects = get_object_store(settings)
    embeddings = HuggingFaceEmbeddingClient(settings)
    ingestion = IngestionService(settings, database, objects, embeddings, ColQwenClient(settings), HuggingFaceASRClient(settings))
    graph = DocMindGraph(database, MemoryService(database), HybridRetriever(database, embeddings), QwenClient(settings), objects)
    return database, ingestion, graph


st.set_page_config(page_title="DocMind", page_icon="🧠", layout="wide")
st.title("🧠 DocMind")
st.caption("Persistent, source-grounded multimodal RAG — hosted inference only")
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
try:
    database, ingestion, graph = services()
except Exception as error:  # noqa: BLE001 -- show deployment configuration failures in the UI
    st.error(f"Configuration error: {error}")
    st.stop()

with st.sidebar:
    st.subheader("This conversation")
    conversation_documents = database.documents(st.session_state.thread_id)
    st.caption(f"{len(conversation_documents)} file(s) available to this chat")
    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

with st.expander("＋ Add files to this chat", expanded=False):
    uploads = st.file_uploader(
        "Upload documents, images, audio, or video",
        accept_multiple_files=True,
        type=["pdf", "docx", "pptx", "xlsx", "txt", "md", "csv", "json", "jpg", "jpeg", "png", "webp", "mp3", "wav", "m4a", "mp4", "mov", "mkv", "webm"],
        key=f"uploads-{st.session_state.thread_id}",
    )
    if uploads and st.button("Add to this chat", key=f"index-{st.session_state.thread_id}"):
        progress = st.progress(0)
        successful_uploads = 0
        for index, upload in enumerate(uploads, start=1):
            try:
                with st.spinner(f"Indexing {upload.name}…"):
                    asyncio.run(ingestion.index_upload(st.session_state.thread_id, upload.name, upload.getvalue(), upload.type))
                successful_uploads += 1
            except Exception as error:  # noqa: BLE001 -- each upload must fail independently
                st.error(f"{upload.name}: {error}")
            progress.progress(index / len(uploads))
        if successful_uploads:
            st.success(f"Added {successful_uploads} file(s) to this chat.")
            st.rerun()
        if successful_uploads != len(uploads):
            st.warning("Failed files were not indexed. Fix the displayed issue, then upload them again.")

for message in database.messages(st.session_state.thread_id, limit=100):
    with st.chat_message(message.role):
        st.write(message.content)
        if message.sources_json != "[]":
            st.caption("Sources: " + message.sources_json)
if question := st.chat_input("Ask about your indexed files"):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"), st.spinner("Retrieving evidence and generating an answer…"):
        try:
            result = asyncio.run(graph.ask(st.session_state.thread_id, question))
            st.write(result["answer"])
            for source in result.get("sources", []):
                st.caption(f"Source: {source.display()}")
        except Exception as error:  # noqa: BLE001 -- hosted services return provider-specific exceptions
            st.error(f"Answer failed: {error}")
