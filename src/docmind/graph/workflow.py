import base64
import mimetypes
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from docmind.core.schemas import Evidence, SourceRef
from docmind.memory.service import MemoryService
from docmind.models.hf import GenerationProvider
from docmind.retrieval.hybrid import HybridRetriever
from docmind.storage.database import Database
from docmind.storage.object_store import ObjectStore


class ChatState(TypedDict, total=False):
    thread_id: str
    question: str
    document_ids: list[str]
    history: str
    evidence: list[Evidence]
    answer: str
    sources: list[SourceRef]


class DocMindGraph:
    def __init__(self, database: Database, memory: MemoryService, retriever: HybridRetriever, generator: GenerationProvider, objects: ObjectStore) -> None:
        self.database, self.memory, self.retriever, self.generator, self.objects = database, memory, retriever, generator, objects
        graph = StateGraph(ChatState)
        graph.add_node("load_memory", self.load_memory)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("generate", self.generate)
        graph.add_node("validate_and_store", self.validate_and_store)
        graph.add_edge(START, "load_memory")
        graph.add_edge("load_memory", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "validate_and_store")
        graph.add_edge("validate_and_store", END)
        self.app = graph.compile()

    def load_memory(self, state: ChatState) -> ChatState:
        messages = self.memory.recent(state["thread_id"])
        return {"history": "\n".join(f"{m.role}: {m.content}" for m in messages)}

    async def retrieve(self, state: ChatState) -> ChatState:
        full_document_request = any(word in state["question"].lower() for word in ("full", "each slide", "all slides", "complete"))
        return {
            "evidence": await self.retriever.retrieve(
                state["question"],
                limit=24 if full_document_request else 8,
                document_ids=state.get("document_ids"),
                thread_id=state["thread_id"],
            )
        }

    async def generate(self, state: ChatState) -> ChatState:
        evidence = state.get("evidence", [])
        if not evidence:
            return {"answer": "I don’t have enough indexed evidence to answer that question.", "sources": []}
        context = f"Recent conversation:\n{state.get('history', '')}\n\n{HybridRetriever.context(evidence)}"
        image_urls = self._visual_urls(evidence)
        answer = await self.generator.answer(state["question"], context, image_urls)
        return {"answer": answer, "sources": [item.source for item in evidence]}

    def _visual_urls(self, evidence: list[Evidence]) -> list[str]:
        """Make selected stored visual evidence available to a hosted VLM without exposing storage publicly."""
        urls: list[str] = []
        for item in evidence:
            if not item.visual_object_key:
                continue
            raw = self.objects.get(item.visual_object_key)
            content_type = mimetypes.guess_type(item.visual_object_key)[0] or "image/jpeg"
            urls.append(f"data:{content_type};base64,{base64.b64encode(raw).decode('ascii')}")
            if len(urls) == 4:
                break
        return urls

    def validate_and_store(self, state: ChatState) -> ChatState:
        self.database.save_message(state["thread_id"], "user", state["question"])
        self.database.save_message(state["thread_id"], "assistant", state["answer"], state.get("sources", []))
        return state

    async def ask(self, thread_id: str, question: str, document_ids: list[str] | None = None) -> ChatState:
        return await self.app.ainvoke({"thread_id": thread_id, "question": question, "document_ids": document_ids or []})
