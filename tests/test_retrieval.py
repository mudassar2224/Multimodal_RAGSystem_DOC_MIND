import pytest

from docmind.core.config import Settings
from docmind.core.schemas import SourceRef
from docmind.retrieval.hybrid import HybridRetriever
from docmind.storage.database import Database


class FakeEmbeddings:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.asyncio
async def test_visual_questions_prioritize_visual_evidence(tmp_path) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'db.sqlite'}", local_object_dir=tmp_path / "objects")
    database = Database(settings)
    database.create_all()
    document_id = database.add_document("chat-a", "clip.mp4", "video", "original/clip", {})
    database.add_chunk(document_id, "A transcript sentence.", SourceRef(document_id, "clip.mp4", kind="transcript"), [1.0, 0.0], None, None)
    database.add_chunk(document_id, "Video frame at 0 seconds", SourceRef(document_id, "clip.mp4", kind="video_frame"), [1.0, 0.0], [1.0, 0.0], "derived/frame.png")

    result = await HybridRetriever(database, FakeEmbeddings()).retrieve("what is happening in the video?", thread_id="chat-a")

    assert result[0].visual_object_key == "derived/frame.png"


@pytest.mark.asyncio
async def test_retrieval_does_not_cross_threads(tmp_path) -> None:
    settings = Settings(database_url=f"sqlite:///{tmp_path / 'db.sqlite'}", local_object_dir=tmp_path / "objects")
    database = Database(settings)
    database.create_all()
    first = database.add_document("chat-a", "a.md", "text", "a", {})
    database.add_chunk(first, "private A", SourceRef(first, "a.md"), [1.0, 0.0], None, None)

    result = await HybridRetriever(database, FakeEmbeddings()).retrieve("private", thread_id="chat-b")

    assert result == []