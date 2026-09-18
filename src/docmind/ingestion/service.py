import hashlib
from pathlib import Path

from docmind.core.config import Settings
from docmind.core.router import detect_modality
from docmind.core.schemas import ChunkDraft, Modality, SourceRef
from docmind.ingestion.chunking import chunk_text
from docmind.ingestion.parsers import (
    extract_video_assets,
    parse_docx,
    parse_pdf,
    parse_pptx,
    parse_tabular,
)
from docmind.models.hf import ASRProvider, EmbeddingProvider, VisualEmbeddingProvider
from docmind.storage.database import Database
from docmind.storage.object_store import ObjectStore


class IngestionService:
    def __init__(self, settings: Settings, database: Database, objects: ObjectStore, embeddings: EmbeddingProvider,
                 visual: VisualEmbeddingProvider, asr: ASRProvider) -> None:
        self.settings, self.database, self.objects = settings, database, objects
        self.embeddings, self.visual, self.asr = embeddings, visual, asr

    async def index_upload(self, thread_id: str, filename: str, data: bytes, mime_type: str | None = None) -> str:
        if len(data) > self.settings.max_upload_mb * 1024 * 1024:
            raise ValueError(f"{filename} exceeds MAX_UPLOAD_MB")
        modality = detect_modality(filename)
        digest = hashlib.sha256(data).hexdigest()
        original_key = self.objects.put(f"original/{digest}/{filename}", data, mime_type)
        document_id = self.database.add_document(thread_id, filename, modality, original_key, {"sha256": digest, "mime_type": mime_type})
        source = SourceRef(document_id=document_id, filename=filename, kind=modality, object_key=original_key)
        try:
            drafts = await self._extract(modality, filename, data, source)
            await self._persist(drafts)
        except Exception:
            self.database.delete_document(document_id)
            raise
        return document_id

    async def _extract(self, modality: Modality, filename: str, data: bytes, source: SourceRef) -> list[ChunkDraft]:
        suffix = Path(filename).suffix.lower()
        if modality is Modality.DOCUMENT and suffix == ".pdf":
            drafts, images = parse_pdf(data, source)
            for key, image in images:
                self.objects.put(key, image, "image/png")
            return drafts
        if modality is Modality.DOCUMENT and suffix == ".docx":
            return parse_docx(data, source)
        if modality is Modality.DOCUMENT and suffix == ".pptx":
            return parse_pptx(data, source)
        if modality is Modality.TEXT:
            return [ChunkDraft(text=x, source=source) for x in chunk_text(data.decode("utf-8", errors="replace"))]
        if modality is Modality.STRUCTURED:
            return parse_tabular(data, suffix, source)
        if modality is Modality.IMAGE:
            return [ChunkDraft(text=f"Image asset: {filename}", source=source, visual_object_key=source.object_key)]
        if modality is Modality.AUDIO:
            return await self._transcript_drafts(data, source, suffix)
        if modality is Modality.VIDEO:
            audio, frames = extract_video_assets(data, suffix, self.settings.video_frame_interval_seconds)
            drafts = await self._transcript_drafts(audio, source, ".wav") if audio else []
            for timestamp, frame in frames:
                key = f"derived/{source.document_id}/frame-{timestamp:.0f}.png"
                self.objects.put(key, frame, "image/png")
                drafts.append(ChunkDraft(text=f"Video frame at {timestamp:.0f} seconds", source=SourceRef(document_id=source.document_id, filename=source.filename, kind="video_frame", start_seconds=timestamp, end_seconds=timestamp, object_key=key), visual_object_key=key))
            return drafts
        raise ValueError(f"No pipeline for {modality}")

    async def _transcript_drafts(self, audio: bytes, source: SourceRef, suffix: str) -> list[ChunkDraft]:
        segments = await self.asr.transcribe(audio, suffix)
        def seconds(value: object | None) -> float | None:
            return float(value) if isinstance(value, (int, float, str)) else None

        return [
            ChunkDraft(
                text=str(segment["text"]),
                source=SourceRef(
                    document_id=source.document_id,
                    filename=source.filename,
                    kind="transcript",
                    start_seconds=seconds(segment.get("start")) or 0,
                    end_seconds=seconds(segment.get("end")),
                ),
            )
            for segment in segments
            if segment.get("text")
        ]

    async def _persist(self, drafts: list[ChunkDraft]) -> None:
        text_drafts = [draft for draft in drafts if draft.text]
        text_vectors = await self.embeddings.embed([draft.text for draft in text_drafts]) if text_drafts else []
        for draft, vector in zip(text_drafts, text_vectors, strict=True):
            visual_vector = None
            if draft.visual_object_key:
                try:
                    visual_vector = await self.visual.embed_image(self.objects.get(draft.visual_object_key))
                except RuntimeError:
                    # Preserve page/frame and its source; visual search is enabled only once a compatible endpoint is configured.
                    visual_vector = None
            self.database.add_chunk(draft.source.document_id, draft.text, draft.source, vector, visual_vector, draft.visual_object_key)
