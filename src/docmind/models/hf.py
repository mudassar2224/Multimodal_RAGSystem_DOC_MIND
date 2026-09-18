import asyncio
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import httpx
from huggingface_hub import InferenceClient
from tenacity import retry, stop_after_attempt, wait_exponential

from docmind.core.config import Settings


class EmbeddingProvider(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class VisualEmbeddingProvider(Protocol):
    async def embed_image(self, image: bytes) -> list[float]: ...


class ASRProvider(Protocol):
    async def transcribe(self, audio: bytes, suffix: str) -> list[dict[str, object]]: ...


class GenerationProvider(Protocol):
    async def answer(self, question: str, context: str, image_urls: list[str]) -> str: ...


class _HFBase:
    def __init__(self, settings: Settings, model: str) -> None:
        if not settings.hf_token:
            raise RuntimeError("HF_TOKEN is required for hosted inference")
        self.token = settings.hf_token
        self.client = InferenceClient(model=model, token=settings.hf_token, provider=settings.hf_provider)


class HuggingFaceEmbeddingClient(_HFBase):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, settings.text_embedding_model)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        result = self.client.feature_extraction(list(texts))
        return result.tolist() if hasattr(result, "tolist") else result

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed, texts)


class ColQwenClient(_HFBase):
    """Adapter boundary for a hosted ColQwen/ColPali compatible endpoint.

    Hugging Face's generic ``feature_extraction`` endpoint accepts text only in
    the installed client version. A provider-specific visual endpoint is needed
    for ColQwen image late interaction, so callers treat its absence as an
    optional visual-index warning rather than losing extracted document text.
    """
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, settings.visual_retrieval_model)
        self.endpoint_url = settings.visual_endpoint_url

    def _embed_image(self, image: bytes) -> list[float]:
        if not self.endpoint_url:
            raise RuntimeError(
                "ColQwen needs a visual endpoint. Set VISUAL_ENDPOINT_URL to a hosted "
                "ColQwen/ColPali service; HF feature_extraction is text-only."
            )
        response = httpx.post(
            self.endpoint_url,
            content=image,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "image/png"},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            payload = payload.get("embedding", payload.get("embeddings"))
        if not isinstance(payload, list) or not payload or not all(isinstance(value, (int, float)) for value in payload):
            raise RuntimeError("The visual endpoint returned an invalid embedding payload.")
        return [float(value) for value in payload]

    async def embed_image(self, image: bytes) -> list[float]:
        return await asyncio.to_thread(self._embed_image, image)


class HuggingFaceASRClient(_HFBase):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, settings.asr_model)

    def _transcribe(self, audio: bytes, suffix: str) -> list[dict[str, object]]:
        # Passing a named file, rather than raw bytes, lets the provider infer a supported audio MIME type.
        descriptor, filename = tempfile.mkstemp(suffix=suffix if suffix.startswith(".") else ".wav")
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(audio)
            result = self.client.automatic_speech_recognition(Path(filename))
            text = result.get("text", "") if isinstance(result, dict) else getattr(result, "text", "")
            return [{"text": text, "start": 0, "end": None}] if text else []
        finally:
            Path(filename).unlink(missing_ok=True)

    async def transcribe(self, audio: bytes, suffix: str) -> list[dict[str, object]]:
        return await asyncio.to_thread(self._transcribe, audio, suffix)


class QwenClient(_HFBase):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings, settings.qwen_model)

    def _answer(self, question: str, context: str, image_urls: list[str]) -> str:
        user_content: list[dict[str, object]] = [{"type": "text", "text": f"Answer only from this evidence. If it is insufficient, say so.\n\nEvidence:\n{context}\n\nQuestion: {question}"}]
        user_content.extend({"type": "image_url", "image_url": {"url": url}} for url in image_urls)
        response = self.client.chat_completion(messages=[{"role": "system", "content": "You are DocMind. Never invent sources."}, {"role": "user", "content": user_content}], max_tokens=700)
        return response.choices[0].message.content or "Evidence was insufficient to answer."

    async def answer(self, question: str, context: str, image_urls: list[str]) -> str:
        return await asyncio.to_thread(self._answer, question, context, image_urls)
