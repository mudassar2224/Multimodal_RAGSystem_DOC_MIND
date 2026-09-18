from pathlib import Path

import pytest

from docmind.core.config import Settings
from docmind.ingestion.service import IngestionService
from docmind.storage.database import Database
from docmind.storage.object_store import LocalObjectStore


class FakeEmbeddings:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]


class FakeVisual:
    async def embed_image(self, image: bytes) -> list[float]:
        return [float(len(image)), 1.0]


class FakeASR:
    async def transcribe(self, audio: bytes, suffix: str) -> list[dict[str, object]]:
        return [{"text": "recorded meeting decision", "start": 3, "end": 9}]


@pytest.mark.asyncio
async def test_text_ingestion_persists_source_metadata(tmp_path: Path) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'db.sqlite'}", local_object_dir=tmp_path / "objects")
    database = Database(settings)
    database.create_all()
    service = IngestionService(settings, database, LocalObjectStore(settings.local_object_dir), FakeEmbeddings(), FakeVisual(), FakeASR())

    document_id = await service.index_upload("thread-1", "report.md", b"Revenue rose by twelve percent in 2024.")

    chunks = database.all_chunks()
    assert len(chunks) == 1
    assert chunks[0].document_id == document_id
    assert "Revenue rose" in chunks[0].text


@pytest.mark.asyncio
async def test_documents_and_chunks_are_isolated_by_thread(tmp_path: Path) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'db.sqlite'}", local_object_dir=tmp_path / "objects")
    database = Database(settings)
    database.create_all()
    service = IngestionService(settings, database, LocalObjectStore(settings.local_object_dir), FakeEmbeddings(), FakeVisual(), FakeASR())

    first = await service.index_upload("chat-a", "a.md", b"Only chat A can see this.")
    second = await service.index_upload("chat-b", "b.md", b"Only chat B can see this.")

    assert [document.id for document in database.documents("chat-a")] == [first]
    assert [chunk.document_id for chunk in database.all_chunks(thread_id="chat-a")] == [first]
    assert [chunk.document_id for chunk in database.all_chunks(thread_id="chat-b")] == [second]
