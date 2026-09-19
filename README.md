<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=220&section=header&text=🧠%20DocMind&fontSize=70&fontColor=ffffff&animation=twinkling&fontAlignY=35&desc=Persistent,%20Source-Grounded%20Multimodal%20RAG&descAlignY=55&descSize=20" width="100%"/>

<br/>

<a href="https://multimodalragsystemdocmind-5fmcbznmemee4llssrmf6n.streamlit.app/">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=26&duration=2800&pause=900&color=6C5CE7&center=true&vCenter=true&multiline=true&repeat=true&width=800&height=100&lines=Ask+questions+across+PDFs%2C+DOCX%2C+PPTX%2C+XLSX...;Images+%F0%9F%96%BC%EF%B8%8F+Audio+%F0%9F%8E%B5+and+Video+%F0%9F%8E%AC+too...;Every+answer+cited.+Every+source+traceable." alt="Typing SVG" />
</a>

<br/><br/>

<!-- Badges -->
<a href="https://multimodalragsystemdocmind-5fmcbznmemee4llssrmf6n.streamlit.app/"><img src="https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" /></a>
<a href="https://github.com/mudassar2224/Multimodal_RAGSystem_DOC_MIND"><img src="https://img.shields.io/badge/💻_Source-GitHub-181717?style=for-the-badge&logo=github&logoColor=white" /></a>
<a href="https://github.com/mudassar2224/Multimodal_RAGSystem_DOC_MIND/stargazers"><img src="https://img.shields.io/github/stars/mudassar2224/Multimodal_RAGSystem_DOC_MIND?style=for-the-badge&color=FFD700&logo=github" /></a>
<a href="https://github.com/mudassar2224/Multimodal_RAGSystem_DOC_MIND/network/members"><img src="https://img.shields.io/github/forks/mudassar2224/Multimodal_RAGSystem_DOC_MIND?style=for-the-badge&color=00C2A8&logo=github" /></a>
<a href="https://github.com/mudassar2224/Multimodal_RAGSystem_DOC_MIND/commits/main"><img src="https://img.shields.io/github/last-commit/mudassar2224/Multimodal_RAGSystem_DOC_MIND?style=for-the-badge&color=6C5CE7" /></a>

<br/>

<img src="https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-1.64-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/LangGraph-1.2-1C3C3C?style=flat-square&logo=langchain&logoColor=white" />
<img src="https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?style=flat-square&logo=postgresql&logoColor=white" />
<img src="https://img.shields.io/badge/Hugging%20Face-Hosted%20Inference-FFD21E?style=flat-square&logo=huggingface&logoColor=black" />
<img src="https://img.shields.io/badge/License-MIT-brightgreen?style=flat-square" />

</div>

<img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="100%">

## ✨ What is DocMind?

> **DocMind** is a persistent, source-grounded **multimodal Retrieval-Augmented Generation (RAG)** system. Drop in documents, spreadsheets, slides, images, audio, or video — ask questions in plain English — and get answers grounded in **cited evidence**, not hallucinated guesses.

Every conversation is a private thread with its own indexed knowledge base. Every answer comes with **page numbers, slide numbers, timestamps, or frame references** pointing straight back to the source.

<div align="center">

### 🎬 See it in action

<img src="https://user-images.githubusercontent.com/74038190/216122041-518ac897-8d92-4c6b-9b3f-ca01dcaf38ee.gif" width="450">

*(Replace this GIF with your own screen recording — see [📸 Adding Your Own Demo](#-adding-your-own-demo) below)*

</div>

---

## 🚀 Live Demo

<div align="center">

### 👉 **[multimodalragsystemdocmind.streamlit.app](https://multimodalragsystemdocmind-5fmcbznmemee4llssrmf6n.streamlit.app/)** 👈

<img src="https://img.shields.io/badge/Status-Live-success?style=for-the-badge&logo=checkmarx" />

</div>

---

## 🌈 Features

| | Feature | Description |
|---|---|---|
| 📄 | **Multi-format ingestion** | PDF, DOCX, PPTX, XLSX, TXT, MD, CSV, JSON |
| 🖼️ | **Image understanding** | Upload screenshots, photos, diagrams — ask what's in them |
| 🎵 | **Audio transcription** | Automatic speech-to-text via Hugging Face ASR |
| 🎬 | **Video comprehension** | Frame extraction + audio transcript, fused for full video Q&A |
| 🔍 | **Hybrid retrieval** | Combines dense text embeddings + visual (ColQwen) retrieval |
| 🕸️ | **LangGraph orchestration** | A real agentic graph workflow, not a single prompt chain |
| 📌 | **Cited, source-grounded answers** | Every answer links back to the exact page / slide / timestamp |
| 🧵 | **Persistent per-thread memory** | Each conversation keeps its own indexed knowledge base |
| ▶️ | **In-app file playback** | Play video/audio, preview PDFs/images/text right inside the chat |
| ☁️ | **Hosted inference only** | No local GPU needed — runs entirely on Hugging Face-hosted models |

---

## 🏗️ Architecture

```mermaid
flowchart LR
    U([👤 User]) -->|Upload files| ST[🖥️ Streamlit UI]
    U -->|Ask question| ST

    ST --> ING[📥 Ingestion Service]
    ING --> EMB[🧬 HF Embeddings]
    ING --> COL[👁️ ColQwen Visual Retrieval]
    ING --> ASR[🎙️ HF ASR]
    ING --> OBJ[(☁️ Object Store)]
    ING --> DB[(🗄️ PostgreSQL + pgvector)]

    ST --> GRAPH{{🕸️ DocMind LangGraph}}
    GRAPH --> RET[🔎 Hybrid Retriever]
    RET --> DB
    GRAPH --> QWEN[🤖 Qwen LLM]
    QWEN --> ANS[✅ Cited Answer]
    ANS --> ST
```

---

## 🧰 Tech Stack

<div align="center">

<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
<img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white"/>
<img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
<img src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white"/>
<img src="https://img.shields.io/badge/pgvector-4169E1?style=for-the-badge&logo=postgresql&logoColor=white"/>
<img src="https://img.shields.io/badge/SQLAlchemy-D71F00?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
<img src="https://img.shields.io/badge/PyMuPDF-EC1C24?style=for-the-badge&logo=adobeacrobatreader&logoColor=white"/>
<img src="https://img.shields.io/badge/AWS%20S3-232F3E?style=for-the-badge&logo=amazons3&logoColor=white"/>
<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
<img src="https://img.shields.io/badge/uv-DE5FE9?style=for-the-badge&logo=python&logoColor=white"/>

</div>

---

## ⚡ Quick Start

```bash
# 1️⃣ Clone the repo
git clone https://github.com/mudassar2224/Multimodal_RAGSystem_DOC_MIND.git
cd Multimodal_RAGSystem_DOC_MIND

# 2️⃣ Install dependencies (uv is used for this project)
uv sync

# 3️⃣ Configure environment variables
cp .env.example .env
# → add your HF_TOKEN, DATABASE_URL, object storage credentials, etc.

# 4️⃣ Run the app
streamlit run app.py
```

<div align="center">
<img src="https://img.shields.io/badge/🌐_Open-http%3A%2F%2Flocalhost%3A8501-6C5CE7?style=for-the-badge"/>
</div>

---

## 🔑 Environment Variables

| Variable | Purpose |
|---|---|
| `HF_TOKEN` | Hugging Face inference API token |
| `QWEN_MODEL` | Qwen LLM model ID (generation) |
| `TEXT_EMBEDDING_MODEL` | Embedding model for hybrid retrieval |
| `VISUAL_RETRIEVAL_MODEL` | ColQwen model for visual retrieval |
| `ASR_MODEL` | Speech-to-text model for audio/video |
| `DATABASE_URL` | PostgreSQL (pgvector) connection string |
| `OBJECT_STORAGE_BACKEND` / `S3_*` | Object storage config for raw file storage |

---

## 📸 Adding Your Own Demo

Want this README to show off *your* actual app instead of placeholder GIFs?

1. Record a short screen capture (e.g. with [ScreenToGif](https://www.screentogif.com/) or [Kap](https://getkap.co/)) of a real Q&A across a PDF + image + audio file.
2. Drop it into a `/docs/assets/` folder in the repo.
3. Replace the placeholder `<img>` tags above with:
   ```md
   ![DocMind demo](docs/assets/your-demo.gif)
   ```

---

## 🗺️ Roadmap

- [x] Multimodal ingestion (PDF, DOCX, PPTX, XLSX, images, audio, video)
- [x] Hybrid text + visual retrieval
- [x] Cited, source-grounded answers
- [x] In-app file preview & playback
- [ ] Legacy `.ppt` / `.doc` / `.xls` ingestion support
- [ ] Multi-user auth & shared workspaces
- [ ] Streaming token-by-token answers

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

```bash
# Fork → branch → commit → PR 🚀
git checkout -b feature/amazing-idea
git commit -m "✨ Add amazing idea"
git push origin feature/amazing-idea
```

---

## 👤 Author

<div align="center">

**Muhammad Mudassar**
Final-year BS Artificial Intelligence @ UMT Lahore · Building toward LLM Engineering

<a href="https://github.com/mudassar2224"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/></a>
<a href="https://linkedin.com/in/muhmmad-mudassar-656384342"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white"/></a>
<a href="https://huggingface.co/Maliktg5"><img src="https://img.shields.io/badge/🤗_Hugging_Face-Maliktg5-FFD21E?style=for-the-badge"/></a>

</div>

---

<div align="center">

### ⭐ If DocMind helped you, consider giving the repo a star!

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer" width="100%"/>

</div>
