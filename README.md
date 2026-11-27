# DocuSense — Full-Stack Cloud RAG & Document QA Platform

[![Live App](<https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white>)](https://docusense-rag.streamlit.app/)
[![FastAPI Backend](<https://img.shields.io/badge/API%20Docs-Swagger%20UI-009688?logo=fastapi&logoColor=white>)](https://docusense-api-nfyq.onrender.com/docs)
[![Backend Health](<https://img.shields.io/badge/Render-Health%20Check-46E3B7?logo=render&logoColor=white>)](https://docusense-api-nfyq.onrender.com/health)

> **DocuSense** is a decoupled, multi-tenant Retrieval-Augmented Generation (RAG) platform engineered to process, index, and query **digital (text-layer) PDF documents**—research papers, financial fact sheets, slide decks, and legal contracts—with **100% citation auditability**, **SSE token streaming**, and **free-tier cloud deployment** under a strict 512 MB RAM ceiling.

- 🌐 **Live Demo UI:** https://docusense-rag.streamlit.app/
- ⚡ **FastAPI Swagger Docs:** https://docusense-api-nfyq.onrender.com/docs
- ❤️ **Backend Health Endpoint:** https://docusense-api-nfyq.onrender.com/health

---

## 🏗️ Decoupled Cloud Architecture (v2.0)

```text
[ User / Recruiter Browser ]
             │
             ▼
[ Streamlit UI — Streamlit Community Cloud ]
   • Cold-start wake-up banner (/health polling)
   • Session UUID persisted in URL query params (?sid=...)
   • 8 free demo queries/session (or custom Groq API key bypass)
             │
             │  HTTPS REST + Server-Sent Events (SSE)
             ▼
[ FastAPI Backend — Render Free Tier (512 MB RAM / 0.1 vCPU) ]
   ├── POST /api/v1/upload        ──► PyMuPDF + Structural Chunker (batch=100)
   ├── POST /api/v1/query/stream  ──► Retrieval + FlashRank + Groq SSE
   ├── POST /api/v1/sessions      ──► Multi-tenant UUID & quota tracking
   └── GET  /health, /memory      ──► Readiness probe & live RSS telemetry
             │                        │                        │
             ▼                        ▼                        ▼
  [ Cloudflare Workers AI ]    [ Qdrant Cloud ]        [ Groq Cloud LPU ]
  @cf/baai/bge-small-en-v1.5   1 GB Persistent         openai/gpt-oss-120b
  (384D GPU, 0 MB local RAM)   (session_id filtered)   (SSE token streaming)
```

---

## 🚀 Key Engineering Highlights

- ☁️ **Zero-PyTorch, 512 MB RAM Backend**: Offloads 384D dense embeddings to **Cloudflare Workers AI** (`@cf/baai/bge-small-en-v1.5` on GPU) and LLM inference to **Groq Cloud**, keeping the FastAPI container lean (~200 MB RSS) with zero PyTorch/Streamlit bloat.
- 🔒 **Multi-Tenant Session Isolation**: Every browser session receives an isolated `session_id` indexed in **Qdrant Cloud** (`MatchValue` payload filtering), ensuring recruiters and concurrent users never see each other's uploaded PDFs.
- ⚡ **Sub-12s Cloud PDF Indexing**: Greedy structural chunk packing + batched Cloudflare embedding (64/batch) + batched Qdrant upsert (100/batch) + fast text-block ingestion (`extract_tables=False` on 0.1 vCPU) cuts 15-page paper indexing from ~158s down to ~10s.
- 🎯 **Two-Stage Retrieval & FlashRank Reranking**: Retrieves `max(top_k × 3, 15)` (21 by default) dense candidates from Qdrant Cloud and reranks them with **FlashRank ONNX** (`ms-marco-TinyBERT-L-2-v2`, pure cross-encoder scoring) down to `TOP_K = 7` chunks before grounded generation.
- 🛡️ **Anti-Abuse Rate Limiting & Custom Key Bypass**: Thread-safe in-memory per-session quota (`8` free queries/session, persisted across browser refreshes via `?sid=`) with instant bypass when a user supplies their own Groq API key (`X-Groq-API-Key`).
- 📜 **Strict Citation Grounding**: Every answer cites exact `[Source X: filename.pdf, Page Y]` references and displays retrieval, reranking, and generation latency telemetry.

---

## 🔌 REST API Reference

Full interactive OpenAPI schema is available at [`/docs`](https://docusense-api-nfyq.onrender.com/docs).

| Method     | Endpoint                                | Purpose                                         |
| :--------- | :-------------------------------------- | :---------------------------------------------- |
| `GET`    | `/health`                             | Readiness probe & vector store status           |
| `GET`    | `/memory`                             | Live process RSS memory & `torch` load check  |
| `POST`   | `/api/v1/sessions`                    | Provision new isolated `session_id` & quota   |
| `GET`    | `/api/v1/sessions/{session_id}/stats` | Session vector count & remaining demo quota     |
| `DELETE` | `/api/v1/sessions/{session_id}`       | Purge session vectors from Qdrant & reset quota |
| `POST`   | `/api/v1/upload`                      | Upload PDF (≤10 MB), chunk, embed & upsert     |
| `POST`   | `/api/v1/query`                       | Full JSON RAG response with citations & timings |
| `POST`   | `/api/v1/query/stream`                | SSE token-streaming RAG endpoint                |

---

## 🏆 RAG Evaluation Scorecard — v2 Cloud Pipeline (40 Test Cases)

> Evaluated with Cloudflare `@cf/baai/bge-small-en-v1.5` embeddings + Qdrant Cloud dense search + FlashRank ONNX reranking + Groq `openai/gpt-oss-120b` generation, judged by local `qwen2.5:7b`. 

| Document Archetype                | Corpus File                       |  Context Precision  |   Context Recall   | Generator Faithfulness | Response Relevancy | Answer Correctness | Status |
| :-------------------------------- | :-------------------------------- | :-----------------: | :-----------------: | :--------------------: | :-----------------: | :-----------------: | :-----: |
| **Narrative Non-Fiction**   | `Atomic Habits.pdf`             | **100.0%** 🏆 | **100.0%** 🏆 |  **100.0%** 🏆  | **100.0%** 🏆 | **100.0%** 🏆 | 🟢 PASS |
| **Academic Paper**          | `Attention is all you need.pdf` | **100.0%** 🏆 | **100.0%** 🏆 |  **100.0%** 🏆  | **100.0%** 🏆 |   **95.0%**   | 🟢 PASS |
| **Slide Deck Presentation** | `Machine Learning PPT.pdf`      | **100.0%** 🏆 | **100.0%** 🏆 |  **100.0%** 🏆  |   **90.0%**   |   **90.0%**   | 🟢 PASS |
| **Financial Fact Sheet**    | `TCS Q4 2025-26 Fact Sheet.pdf` |   **90.0%**   | **100.0%** 🏆 |  **100.0%** 🏆  | **100.0%** 🏆 |   **87.0%**   | 🟢 PASS |
| **OVERALL (40 TCs)**        |                                   |   **97.5%**   | **100.0%** 🏆 |  **100.0%** 🏆  |   **97.5%**   |   **93.0%**   |   🟢   |

---

## 🛠️ Local Development Setup

### Prerequisites

- **Python**: 3.10+ (always run inside `.venv`)
- **Cloud Accounts (Free Tier)**: [Qdrant Cloud](https://cloud.qdrant.io/), [Cloudflare Workers AI](https://dash.cloudflare.com/), [Groq](https://console.groq.com/)

### 1. Install Dependencies

```powershell
# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install backend + frontend dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements-backend.txt -r requirements.txt
```

### 2. Configure `.env`

Copy `.env.example` to `.env` and fill in your keys:

```env
QDRANT_URL=https://your-cluster-url.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=docusense_docs
VECTOR_DIMENSION=384

CF_ACCOUNT_ID=your_cloudflare_account_id
CF_API_TOKEN=your_cloudflare_api_token

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

### 3. Run Backend & Frontend

```powershell
# Terminal 1 — FastAPI Backend
.\.venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — Streamlit Frontend
.\.venv\Scripts\streamlit.exe run ui/streamlit_app.py
```

Open `http://localhost:8501` for the UI and `http://localhost:8000/docs` for Swagger API docs.

---

## ⚠️ Known Limitations (Free-Tier Cloud)

- **No Cloud OCR**: Image-only / scanned PDFs without a text layer are rejected (`422`) to keep the Render container under 512 MB RAM.
- **Render 15-Min Sleep**: The free-tier backend sleeps after 15 minutes of idle time (~45–60s cold-start handled automatically by the UI banner).
- **10 MB Upload Cap & 8-Query Demo Limit**: Bypassable for queries by entering your own Groq API key in the sidebar.

---

## 📄 License

Developed as an AI engineering project for DocuSense. Licensed under the MIT License.
