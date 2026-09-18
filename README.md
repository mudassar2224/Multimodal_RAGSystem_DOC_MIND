# DocMind

DocMind is a fully hosted, persistent multimodal RAG application. Streamlit runs the UI; inference is delegated to configured Hugging Face Inference Providers. It never loads a model locally and does not need a GPU.

## What is implemented

The baseline routes PDFs, DOCX, PPTX, XLSX, text/structured files, images, audio, and video; stores originals and derived page/frame images; preserves citations as metadata; uses hosted adapters for text embeddings, ColQwen visual embeddings, Whisper ASR, and Qwen3-VL generation; and persists files, chunks, evidence and conversation history.

PDF pages and PPTX slides are rendered as images. Video extraction requires `ffmpeg` on the **server** (CPU-only is fine). For production set PostgreSQL with `pgvector` and an S3-compatible object store. SQLite/local files are deliberately provided only for simple local development and tests.

## Local development

1. Install [uv](https://docs.astral.sh/uv/), then run `uv sync --all-groups`.
2. Copy `.env.example` to `.env`, add `HF_TOKEN`, and choose hosted models/providers compatible with your account.
3. Run `uv run streamlit run ui/streamlit_app.py`.
4. Verify with `uv run pytest`, `uv run ruff check .`, and `uv run mypy src`.

No secret belongs in Git. Set the same values in Streamlit secrets or deployment environment variables.

## Deployment

Deploy Streamlit with `DATABASE_URL` pointing to managed PostgreSQL (with `CREATE EXTENSION vector`) and `OBJECT_STORAGE_BACKEND=s3` plus S3 credentials. Install `ffmpeg` in the image when video ingestion is enabled. The Streamlit process is the web/UI tier; for larger files move `IngestionService.index_upload` into a worker queue before scaling it horizontally.

## Provider configuration

Inference APIs differ by task. The `models/` adapters isolate that variance: `QwenClient`, `EmbeddingClient`, `ASRClient`, and `ColQwenClient`. Qwen, text embeddings, and Whisper use Hugging Face hosted inference. ColQwen image embeddings require a compatible hosted visual endpoint configured with `VISUAL_ENDPOINT_URL`; Hugging Face's generic `feature_extraction` route is text-oriented and cannot accept ColQwen image inputs. Without that endpoint, text/transcript retrieval still works and visual frames remain available to Qwen when a visual question is asked.
