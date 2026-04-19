# Sankofa AI — Ghana Elections & Budget Intelligence

**Student:** Eli Afi Ayekpley
**Index Number:** 10012200058
**Course:** CS4241 - Introduction to Artificial Intelligence
**Institution:** Academic City University College
**Year:** 2026

A full Retrieval-Augmented Generation (RAG) application built for the CS4241 AI exam. The system answers questions about Ghana Election Results and the 2025 Ghana Budget Statement using vector search + a large language model, with no LangChain or LlamaIndex.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                           │
│                     Next.js Frontend                            │
│   [Chat UI] [Sources Panel] [Prompt Viewer] [Compare Mode]      │
└──────────────────────────┬──────────────────────────────────────┘
                           │  HTTP (REST)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Flask Backend (app.py)                        │
│  POST /api/chat   POST /api/compare   POST /api/feedback         │
│  GET  /health     GET  /api/logs      GET  /api/feedback/stats   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌─────────────────────┐    ┌────────────────────────┐
│  EmbeddingPipeline  │    │     RAGPipeline         │
│  SentenceTransformer│    │  build_prompt_v3()      │
│  all-MiniLM-L6-v2   │    │  context window mgmt   │
│  L2 normalisation   │    │  hallucination control  │
└────────┬────────────┘    └──────────┬─────────────┘
         │                            │
         ▼                            ▼
┌─────────────────────┐    ┌────────────────────────┐
│    VectorStore      │    │      Groq API           │
│  FAISS IndexFlatIP  │    │  llama-3.3-70b-versatile│
│  cosine similarity  │    │  (LLM generation)       │
│  top-k retrieval    │    └────────────────────────┘
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                  Data Layer                          │
│  faiss.index   chunks.json   feedback.json   logs/   │
│  (1002 vectors from CSV + PDF)                       │
└─────────────────────────────────────────────────────┘
         ▲
         │  (built once by setup.py)
┌─────────────────────────────────────────────────────┐
│               data_ingestion.py                      │
│  Ghana_Election_Result.csv  →  row-based chunks      │
│                             +  national summaries    │
│  2025_Budget_Statement.pdf  →  sliding window chunks │
└─────────────────────────────────────────────────────┘
```

**Data flow for a user query:**
1. User types question → frontend sends `{question, retrieval_query}` to `/api/chat`
2. `retrieval_query` (clean question) → EmbeddingPipeline → 384-dim L2-normalised vector
3. FAISS `IndexFlatIP` finds top-10 nearest vectors (cosine similarity)
4. Retrieved chunks ranked by score, truncated at 3000-token budget → `build_prompt_v3()`
5. Full contextual question + injected context → Groq LLM → answer
6. Response logged to `logs/`, returned to frontend with sources + prompt

**Why this design suits the domain:**
- Election CSV data is structured (per-region rows) → row-based chunking preserves atomic facts
- Budget PDF is dense policy prose → sliding window prevents context loss at boundaries
- FAISS IndexFlatIP is sufficient for ~1000 vectors (no need for approximate search)
- Groq's llama-3.3-70b provides strong reasoning for aggregating regional vote data

---

## Part A: Chunking Strategy & Justification

### CSV — Row-based chunking
Each CSV row (one region × one candidate × one election year) becomes one chunk. Rows are **atomic units** — splitting mid-row would destroy the relationship between candidate, party, votes, and region.

### PDF — Sliding window (500 words, 50-word overlap)
| Parameter | Value | Justification |
|-----------|-------|---------------|
| Chunk size | 500 words | Fits within embedding model's 256-token effective range (~400 words) while keeping enough policy context for semantic coherence |
| Overlap | 50 words (10%) | Prevents retrieval misses at chunk boundaries where a policy statement spans two chunks |

### Chunking Impact on Retrieval Quality
| Strategy | Query: "Who won 2020 election?" | Query: "Budget highlights" |
|----------|--------------------------------|--------------------------|
| Row-only CSV | ✅ Finds regional rows but LLM can't aggregate | ✅ Finds budget sections |
| Row + national summaries | ✅ Summary chunk retrieved directly (score 0.71) | ✅ |
| PDF sliding window (500/50) | Irrelevant budget tables retrieved | ✅ Policy text preserved |
| PDF page-based (no overlap) | Same | ❌ Sentences split at page boundaries cause incomplete context |

**Fix implemented:** National summary chunks were added to the CSV index to handle broad queries like "highlights" or "who won" — these aggregate per-region data into a single searchable document per election year.

---

## Part C: Prompt Engineering Iterations

### v1 — Basic instruction
```
Answer the question using the context below.
Context: {context}
Question: {question}
```
**Problem:** LLM fabricated vote totals not in context.

### v2 — Hallucination guard added
```
Answer ONLY using the provided context. If insufficient, say so.
Rules: cite sources, no fabrication.
```
**Problem:** For election questions, LLM said "insufficient context" even when regional rows were retrieved, because it couldn't aggregate them.

### v3 (current) — Domain-aware aggregation instruction
```
You are a precise research assistant for Ghana election and budget data.
For election questions: identify the candidate with the most votes across
the regions shown. If only partial data is shown, note that.
If context is truly irrelevant, say: "I don't have enough context."
```
**Improvement:** LLM now correctly aggregates regional CSV rows and answers election questions. Hallucination control preserved. Explicitly handles the partial-data case.

---

## Part E: Adversarial Testing

### Query 1 — Ambiguous (no year specified)
**Query:** "Who won the Ghana presidential election?"
- **RAG:** Retrieves most recent year's summary (2020), answers correctly
- **Pure LLM:** Answers about 2020 from training data — may hallucinate exact vote counts
- **Finding:** RAG is grounded to dataset; LLM may add unverified statistics

### Query 2 — Misleading (false premise)
**Query:** "I thought John Dramani Mahama won the 2020 election"
- **RAG:** Retrieves 2020 summary chunk, correctly identifies Akufo-Addo as winner, does not validate the false premise
- **Pure LLM:** Correctly corrects the user using training knowledge
- **Finding:** RAG performs well here; both systems handle false premises correctly. LLM latency ~0.8s vs RAG ~3s (retrieval overhead).

---

## Part G: Innovation — Feedback Loop

Users can rate every response 👍 / 👎. Ratings are stored in `logs/feedback.json` and aggregated via `/api/feedback/stats`. This creates a feedback signal for future fine-tuning or retrieval reranking.

Additionally: **query expansion** generates 2 rephrased variants of each query, retrieves for all 3, deduplicates and re-ranks — improving recall for ambiguous queries (toggle via `use_expansion=true`).

---

## Setup Instructions

### Prerequisites
- Python 3.10+, Node.js 18+
- Free Groq API key from [console.groq.com](https://console.groq.com)

### Backend
```bash
cd ai_10012200058/backend
pip install -r requirements.txt
cp .env.example .env          # add your GROQ_API_KEY
python setup.py               # downloads data + builds FAISS index
python app.py                 # starts on http://localhost:5000
```

### Frontend
```bash
cd ai_10012200058/frontend
npm install
npm run dev                   # starts on http://localhost:3000
```

---

## Deployment

### Backend → Render
1. Push to GitHub repo named `ai_10012200058`
2. Connect repo to [render.com](https://render.com) — `render.yaml` is pre-configured
3. Set `GROQ_API_KEY` in Render environment variables

### Frontend → Vercel
1. Import repo to [vercel.com](https://vercel.com), set root to `frontend/`
2. Set env var: `NEXT_PUBLIC_API_URL=https://your-render-url.onrender.com`

---

## Exam Parts Coverage

| Part | Marks | Feature | File |
|------|-------|---------|------|
| A | 4 | Data cleaning, row-based CSV chunking, PDF sliding window 500/50 | `data_ingestion.py` |
| B | 6 | SentenceTransformer embeddings, FAISS IndexFlatIP, top-k, query expansion, failure case + fix | `embeddings.py` |
| C | 4 | Prompt v1→v3 iterations, context window management, hallucination control | `rag_pipeline.py` |
| D | 10 | Full pipeline with stage logging, retrieved docs + scores + prompt displayed in UI | `rag_pipeline.py`, `page.tsx` |
| E | 6 | 2 adversarial queries, RAG vs pure LLM compare mode with evidence | `rag_pipeline.py`, `page.tsx` |
| F | 8 | Architecture diagram above, data flow, domain justification | This README |
| G | 6 | Feedback loop (thumbs up/down → stats), query expansion | `rag_pipeline.py`, `page.tsx` |
| Deliverables | 16 | UI, GitHub, deployment, video, experiment logs, documentation | All |
