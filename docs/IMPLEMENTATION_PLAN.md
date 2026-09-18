# DocMind delivery plan

This repository is a clean implementation; the linked GitHub repository was empty when inspected. The order below is intentional: every phase has a testable exit condition before the next introduces operational risk.

## 1. Foundation — complete

- uv project, lockfile, typed configuration and secret template.
- File routing, persistent document/chunk/message schema, local-development object store and S3 adapter.
- Hosted provider interfaces; no local model loading and no GPU dependency.
- Unit/integration tests use fakes, so they do not consume a Hugging Face token.

Exit gate: `uv sync --all-groups && uv run pytest && uv run ruff check . && uv run mypy src --follow-imports=skip`.

## 2. Production storage before public deployment

- Provision managed PostgreSQL, enable `vector`, migrate vector fields from the development JSON fallback to pgvector columns and add HNSW/IVFFlat indexes.
- Provision an S3-compatible bucket, set lifecycle rules and switch `OBJECT_STORAGE_BACKEND=s3`.
- Add Alembic migrations and a worker queue for indexing; Streamlit must not process large uploads in its request process.
- Add authentication, tenant/user IDs, file-size quotas, malware scanning and signed object URLs.

Exit gate: restart the services, then verify document, chunk, message and source records remain retrievable from PostgreSQL and originals/derived media remain retrievable from object storage.

## 3. Retrieval correctness

- Confirm the chosen Hugging Face provider exposes each required task and model: Qwen3-VL chat completion, text embeddings, Whisper ASR and a ColQwen/ColPali-compatible visual endpoint.
- Persist native ColQwen late-interaction representations or a provider-supported search index; do not treat Qwen as a retriever.
- Add lexical/BM25 retrieval, modality-aware score calibration, a hosted reranker, duplicate suppression and query-mode selection.
- Create a golden evaluation dataset containing citation expectations for PDF pages, XLSX ranges, audio timestamps and video frames.

Exit gate: evaluate recall@k, citation precision and unsupported-answer rate on that dataset before changing prompts or models.

## 4. Media ingestion

- Install `ffmpeg` in the deployment image and validate audio extraction, frame sampling and timestamp alignment on representative MP4/MOV inputs.
- Install a server-side LibreOffice/unoconv renderer (or an equivalent cloud render service) to convert PPTX slides to page images before ColQwen indexing; text extraction alone is not sufficient for visually rich slides.
- Add OCR where scanned pages/images need searchable text, and test corrupt/password-protected/unsupported files separately.

Exit gate: each required extension has one successful fixture, one failure fixture, preserved source metadata and a displayed citation.

## 5. Conversation and UX

- Use a LangGraph durable checkpointer backed by PostgreSQL, with recent messages plus summary/long-term memory—not full history in every prompt.
- Render citations as structured links/cards and show image/page/frame previews using short-lived object URLs.
- Add cancellation/progress events from the worker, retry status, document deletion/reindexing and clear provider-configuration diagnostics.

Exit gate: a fresh browser session can reopen an old thread, cite its evidence exactly and preview the cited media.

## 6. Deployment gate

- Configure all secrets in the deployment environment/Streamlit secrets; never in Git.
- Run integration tests against a disposable PostgreSQL+pgvector bucket and mocked HTTP provider failures.
- Add CI for format, typing, tests, dependency audit and secret scanning, then deploy staging before production.

## Required configuration

Copy `.env.example` to `.env`. `HF_TOKEN` is mandatory for real ingestion/chat. SQLite/local storage are only developer fallbacks; they are intentionally not a production deployment configuration.
