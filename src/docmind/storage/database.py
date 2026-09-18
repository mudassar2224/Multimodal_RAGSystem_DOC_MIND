import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, ForeignKey, String, Text, create_engine, inspect, select, text
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from docmind.core.config import Settings
from docmind.core.schemas import SourceRef


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    thread_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    filename: Mapped[str] = mapped_column(String(512))
    modality: Mapped[str] = mapped_column(String(32))
    object_key: Mapped[str] = mapped_column(String(1024))
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    text: Mapped[str] = mapped_column(Text)
    source_json: Mapped[str] = mapped_column(Text)
    text_embedding_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    visual_embedding_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    visual_object_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    document: Mapped[Document] = relationship(back_populates="chunks")


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    thread_id: Mapped[str] = mapped_column(String(128), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    sources_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Database:
    def __init__(self, settings: Settings) -> None:
        if settings.database_url.startswith("sqlite:///"):
            database_path = settings.database_url.removeprefix("sqlite:///")
            if database_path not in {":memory:", ""}:
                Path(database_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
        self.engine = create_engine(settings.database_url, connect_args=connect_args)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        Base.metadata.create_all(self.engine)
        # Lightweight development migration. Production uses migrations before deployment.
        columns = {column["name"] for column in inspect(self.engine).get_columns("documents")}
        if "thread_id" not in columns:
            with self.engine.begin() as connection:
                connection.execute(text("ALTER TABLE documents ADD COLUMN thread_id VARCHAR(128)"))
                connection.execute(text("UPDATE documents SET thread_id = 'legacy' WHERE thread_id IS NULL"))

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self.sessions() as session:
            yield session
            session.commit()

    def add_document(self, thread_id: str, filename: str, modality: str, object_key: str, metadata: dict[str, object]) -> str:
        document_id = str(uuid.uuid4())
        with self.session() as s:
            s.add(Document(id=document_id, thread_id=thread_id, filename=filename, modality=modality, object_key=object_key,
                           metadata_json=json.dumps(metadata), created_at=datetime.now(UTC)))
        return document_id

    def add_chunk(self, document_id: str, text: str, source: SourceRef, text_embedding: list[float] | None,
                  visual_embedding: list[float] | None, visual_object_key: str | None) -> None:
        with self.session() as s:
            s.add(Chunk(id=str(uuid.uuid4()), document_id=document_id, text=text,
                        source_json=json.dumps(asdict(source)), text_embedding_json=json.dumps(text_embedding) if text_embedding else None,
                        visual_embedding_json=json.dumps(visual_embedding) if visual_embedding else None,
                        visual_object_key=visual_object_key))

    def all_chunks(self, document_ids: list[str] | None = None, thread_id: str | None = None) -> list[Chunk]:
        with self.session() as s:
            statement = select(Chunk).join(Chunk.document)
            if document_ids:
                statement = statement.where(Chunk.document_id.in_(document_ids))
            if thread_id is not None:
                statement = statement.where(Document.thread_id == thread_id)
            return list(s.scalars(statement).all())

    def delete_document(self, document_id: str) -> None:
        with self.session() as s:
            document = s.get(Document, document_id)
            if document:
                s.delete(document)

    def save_message(self, thread_id: str, role: str, content: str, sources: list[SourceRef] | None = None) -> None:
        with self.session() as s:
            s.add(Message(id=str(uuid.uuid4()), thread_id=thread_id, role=role, content=content,
                          sources_json=json.dumps([asdict(x) for x in sources or []]), created_at=datetime.now(UTC)))

    def messages(self, thread_id: str, limit: int = 12) -> list[Message]:
        with self.session() as s:
            rows = list(s.scalars(select(Message).where(Message.thread_id == thread_id).order_by(Message.created_at.desc()).limit(limit)))
            return list(reversed(rows))

    def documents(self, thread_id: str | None = None) -> list[Document]:
        with self.session() as s:
            statement = select(Document).order_by(Document.created_at.desc())
            if thread_id is not None:
                statement = statement.where(Document.thread_id == thread_id)
            return list(s.scalars(statement).all())
