import json
import math

from docmind.core.schemas import Evidence, SourceRef
from docmind.models.hf import EmbeddingProvider
from docmind.storage.database import Database


def cosine(a: list[float], b: list[float]) -> float:
    denominator = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b, strict=False)) / denominator if denominator else 0.0


class HybridRetriever:
    def __init__(self, database: Database, embeddings: EmbeddingProvider) -> None:
        self.database, self.embeddings = database, embeddings

    async def retrieve(
        self,
        query: str,
        limit: int = 8,
        document_ids: list[str] | None = None,
        thread_id: str | None = None,
    ) -> list[Evidence]:
        query_vector = (await self.embeddings.embed([query]))[0]
        ranked: list[Evidence] = []
        for chunk in self.database.all_chunks(document_ids, thread_id=thread_id):
            if not chunk.text_embedding_json:
                continue
            score = cosine(query_vector, json.loads(chunk.text_embedding_json))
            ranked.append(Evidence(chunk_id=chunk.id, text=chunk.text, source=SourceRef(**json.loads(chunk.source_json)), score=score, visual_object_key=chunk.visual_object_key))
        # Generic text embeddings cannot understand a frame label such as
        # "Video frame at 30 seconds". For visual questions, keep visual
        # chunks available so the VLM can inspect the actual frame.
        visual_query = any(word in query.lower() for word in ("video", "image", "photo", "show", "happen", "look", "see", "visual"))
        ordered = sorted(ranked, key=lambda item: item.score, reverse=True)
        if visual_query:
            visual = [item for item in ordered if item.visual_object_key]
            ordered = visual + [item for item in ordered if not item.visual_object_key]

        unique: list[Evidence] = []
        seen_sources: set[str] = set()
        for item in ordered:
            identity = item.source.display()
            if identity not in seen_sources:
                unique.append(item)
                seen_sources.add(identity)
            if len(unique) == limit:
                break
        return unique

    @staticmethod
    def context(evidence: list[Evidence]) -> str:
        return "\n\n".join(f"[{index}] {item.text}\nSOURCE: {item.source.display()}" for index, item in enumerate(evidence, start=1))
