# Vector Search Service

A **production-ready FastAPI-based vector search system** that enables semantic search over documents (PDFs and images) using OCR, intelligent chunking, and vector embeddings stored in ChromaDB — with hybrid retrieval, cross-encoder reranking, and a citation-backed RAG endpoint on top.

![demo](docs/demo.gif)
*(record a quick screen capture of `/docs` — index a PDF, run a hybrid+rerank search, then hit `/vector/ask` — and drop it at `docs/demo.gif`)*

---

## Quickstart (Docker Compose)

**Recommended:** Both backend API and Streamlit frontend in containers — no local Python setup needed.

```bash
docker-compose up
```

Open **http://localhost:8501** for the Streamlit frontend. The backend API runs on **http://localhost:8000** (Swagger UI at `/docs`).

**Optional:** To enable `/vector/ask` (RAG with Gemini), edit `docker-compose.yml` and uncomment `GOOGLE_API_KEY`, or set it inline:

```bash
docker-compose run -e GOOGLE_API_KEY=your_key backend
```

Get a free key from [Google AI Studio](https://aistudio.google.com/apikey) — no billing required.

---

## 🎯 Overview

This service ingests documents, extracts text using OCR, splits the text into semantically meaningful chunks, converts them into vector embeddings, and stores them in a vector database for **fast, semantic similarity search**.

Unlike keyword search, this system retrieves results based on **meaning and context**, not exact word matches. On top of retrieval, it adds a full modern RAG pipeline: **hybrid dense+keyword search → cross-encoder reranking → cited answer generation**.

---

## ✨ Key Features

* **OCR Text Extraction**
  Extracts text from PDFs and images using Tesseract OCR.

* **Configurable Chunking**
  Chunk size, overlap, and strategy are configurable to balance recall, precision, and cost.

* **Semantic Embeddings**
  Uses Sentence Transformers to generate dense vector representations (384 dimensions).

* **Approximate Nearest Neighbor (ANN) Search**
  Uses ChromaDB's HNSW-style index for low-latency similarity search.

* **Hybrid Search (Dense + BM25)**
  Optionally fuses semantic (embedding) search with keyword (BM25) search via Reciprocal Rank Fusion — catches exact terms, IDs, and rare proper nouns that embeddings alone can miss.

* **Cross-Encoder Reranking**
  Optionally re-scores retrieved candidates with a cross-encoder for a second, more precise relevance pass before returning results.

* **RAG with Citations**
  `/vector/ask` retrieves context and generates an answer via Google AI Studio's free-tier Gemini API, with every claim cited back to its source chunk.

* **Metadata Filtering**
  Supports filtering by `user_id`, `document_id`, `document_name`, `tags`, and custom metadata.

* **Persistent Storage**
  Uses DuckDB + Parquet for durable, disk-backed vector storage.

---

## 🔄 System Architecture

### Document Indexing Flow

```
PDF / Image
   ↓
OCRService.extract_text()
   ↓
ChunkingService.chunk_text()
   ↓
EmbeddingService.embed()
   ↓
ChromaRepository.add()
```

**Explanation:**

1. OCR converts the document into raw text
2. Text is split into overlapping chunks
3. Each chunk is embedded into a dense vector
4. Vectors + metadata are stored in ChromaDB

---

### Search Flow

```
Query Text
   ↓
EmbeddingService.embed()  ──────────┐
   ↓                                │  hybrid=true
ChromaRepository.search()   BM25Okapi over candidate pool
   ↓                                │
   └───────────► Reciprocal Rank Fusion ◄┘
                        ↓
              Metadata / Tag Post-Filtering
                        ↓
        (rerank=true) CrossEncoder re-scoring
                        ↓
                  Top-K Results
```

`hybrid` and `rerank` are opt-in request flags — the default path is plain dense search + metadata filtering, unchanged from the diagram above minus the two optional branches. See [Retrieval Pipeline](#-retrieval-pipeline-hybrid-search--reranking) below for how they work.

---

## 🚀 API Endpoints

### 1️⃣ POST `/vector/index`

Index a document for semantic search.

**Request**

```bash
curl -X POST http://localhost:8000/vector/index \
  -F "file=@document.pdf" \
  -F "user_id=user123" \
  -F "tags=medical,urgent"
```

**Response**

```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "total_chunks": 45
}
```

---

### 2️⃣ POST `/vector/search`

Search indexed documents using semantic similarity — optionally hybrid (dense + BM25) and/or cross-encoder reranked.

**Request**

```json
{
  "query": "What are the symptoms of hypertension?",
  "top_k": 5,
  "filters": {
    "user_id": "user123",
    "tags": ["medical"]
  },
  "hybrid": true,
  "rerank": true
}
```

`hybrid` and `rerank` both default to `false` (existing dense-search behavior). Set either or both to opt into the fuller retrieval pipeline — see [Retrieval Pipeline](#-retrieval-pipeline-hybrid-search--reranking).

**Response**

```json
{
  "ids": [...],
  "distances": [...],
  "documents": [...],
  "metadatas": [...],
  "rerank_scores": [...]
}
```

`rerank_scores` is only present when `rerank: true`.

---

### 3️⃣ GET `/vector/stats`

Returns index-level statistics.

```json
{
  "total_chunks": 1024
}
```

---

### 4️⃣ POST `/vector/ask`

Retrieval-augmented generation: retrieves context (dense, hybrid, and/or reranked — same flags as `/vector/search`) and asks Gemini to answer strictly from that context, citing sources inline. Powered by [Google AI Studio](https://aistudio.google.com/apikey)'s free-tier Gemini API — requires `GOOGLE_API_KEY` in `.env`.

**Request**

```json
{
  "query": "What are the symptoms of hypertension?",
  "top_k": 5,
  "hybrid": true,
  "rerank": true,
  "filters": {
    "user_id": "user123"
  }
}
```

**Response**

```json
{
  "answer": "Common symptoms include headaches and dizziness [1], though many cases are asymptomatic [2].",
  "citations": [
    {
      "index": 1,
      "document_id": "550e8400-...",
      "document_name": "Cardiology Notes.pdf",
      "page_number": 3,
      "chunk_text": "...",
      "distance": 0.18
    },
    {
      "index": 2,
      "document_id": "550e8400-...",
      "document_name": "Cardiology Notes.pdf",
      "page_number": 5,
      "chunk_text": "...",
      "distance": 0.24
    }
  ]
}
```

If `GOOGLE_API_KEY` isn't set, this returns `503` with instructions rather than failing with a raw SDK error.

---

## Retrieval Pipeline: Hybrid Search & Reranking

**Hybrid search** fuses two independent rankings of the same query:

- **Dense**: the existing embedding similarity search (semantic meaning).
- **Sparse**: [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) keyword scoring over the (optionally metadata-filtered) candidate pool — catches exact terms, codes, and rare proper nouns that embeddings can blur together.

The two rankings are combined with **Reciprocal Rank Fusion** (`score(d) = Σ 1/(k + rank)`, `k=60`), which avoids having to tune a weighted blend of two differently-scaled scores (cosine distance vs. BM25 score) — a document's *rank* in each list matters, not the raw score.

*Trade-off:* the BM25 index is rebuilt in-memory from the filtered candidate pool on every hybrid query — O(pool size) per request. Fine at portfolio scale (thousands of chunks); a production system would maintain a persistent/incremental sparse index (Elasticsearch, Typesense, tantivy) instead of rebuilding one per query.

**Cross-encoder reranking** adds a second, more expensive pass: a bi-encoder (used for the dense search above) embeds the query and each document independently, which is fast but leaves some relevance signal on the table. A cross-encoder scores the query and a candidate *together* in one forward pass — far more accurate, but too slow to run over an entire collection. So when `rerank: true`, retrieval first over-fetches `top_k × RERANK_OVERFETCH` candidates (via dense or hybrid search), then the cross-encoder re-scores and narrows back to `top_k`. Reranking a pool already trimmed to `top_k` could only reorder it, not surface better candidates ranked just outside the first-pass cutoff — the overfetch is what makes reranking worth doing.

`/vector/ask` reuses this exact same pipeline for its context, so improving retrieval quality (via `hybrid`/`rerank`) improves answer quality too — it's one retrieval stack behind both endpoints, not a separate implementation.

---

## ⚙️ Configuration Highlights

| Parameter          | Purpose                                    | Typical Value                         |
| ------------------ | ------------------------------------------- | -------------------------------------- |
| `chunk_size`       | Size of each text chunk                     | 400–800 chars                          |
| `chunk_overlap`    | Overlap between chunks                      | 50–150 chars                           |
| `embedding_dim`    | Vector dimension                            | 384                                    |
| `top_k`            | Search results returned                     | 5–20                                   |
| `RERANK_MODEL`     | Cross-encoder used when `rerank: true`      | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `RERANK_OVERFETCH` | Candidate-pool multiplier before reranking  | 4                                      |
| `GEMINI_MODEL`     | Model used by `/vector/ask`                 | `gemini-2.5-flash`                     |

---

## 🧠 Index Type & Search Algorithm

### Index Type: **HNSW (Hierarchical Navigable Small World)**

ChromaDB internally uses an **HNSW-style ANN index**, a graph-based structure optimized for fast similarity search.

#### Why HNSW?

* Logarithmic search complexity
* Excellent recall (95–99%)
* Very fast query times at scale

#### Conceptual Structure

```
Layer 2 (Sparse, global)
Layer 1 (Medium density)
Layer 0 (Dense, local)

Search starts at top → drills down
```

**Trade-off:**

* Extremely fast
* Approximate (not exact KNN)

---

## ⏱️ Time Complexity Analysis

### Indexing (Per Document)

| Stage       | Complexity | Notes                |
| ----------- | ---------- | -------------------- |
| OCR         | O(p)       | p = number of pixels |
| Chunking    | O(n)       | n = text length      |
| Embedding   | O(c × d)   | c = chunks, d = 384  |
| HNSW Insert | O(log² N)  | N = total vectors    |
| Persistence | O(N)       | Disk write           |

**Total:** `O(n + c·d + log²N)`

---

### Searching

| Stage           | Complexity   | Notes              |
| --------------- | ------------ | ------------------ |
| Query embedding | O(q × d)     | q = query length   |
| ANN traversal   | O(log N × k) | k = top_k          |
| Metadata filter | O(k)         | Simple comparisons |

**Typical latency:** **5–50 ms**

---

## 🧩 Metadata Filtering Strategy

### Strategy Used: **Post-Filtering**

```
Vector Search → Candidate Set → Metadata Filter → Top-K
```

#### Why Post-Filtering?

| Approach    | Pros        | Cons                      |
| ----------- | ----------- | ------------------------- |
| Pre-filter  | Faster      | May miss relevant vectors |
| Post-filter | Best recall | Slight overhead           |
| Hybrid      | Best of both, and now implemented (see [Retrieval Pipeline](#-retrieval-pipeline-hybrid-search--reranking)) | Dense + sparse rankings must be fused, adding a query-time cost |

**Current choice:** Post-filtering for metadata, opt-in dense+BM25 hybrid fusion for retrieval itself.

---

### Filtering Performance

* Equality checks (`user_id`, `document_id`): **O(k)**
* Tag matching (list/string): **O(k)**
* Typical cost: **<1 ms**

---

## 🧠 Memory vs Speed Trade-offs

### 1️⃣ Chunk Size

| Small Chunks     | Large Chunks     |
| ---------------- | ---------------- |
| Better precision | Better context   |
| More embeddings  | Fewer embeddings |
| Higher memory    | Lower memory     |
| Slower indexing  | Faster indexing  |

**Rule of Thumb:**

* QA systems → smaller chunks
* Summarization → larger chunks

---

### 2️⃣ Persistence Mode

| In-Memory     | Disk-Persisted  |
| ------------- | --------------- |
| Fastest       | Slightly slower |
| No durability | Crash-safe      |
| RAM-bound     | Scales to TB    |

**Current choice:** Disk-backed (DuckDB + Parquet)

---

### 3️⃣ Search Quality vs Speed

| top_k | Quality     | Latency |
| ----- | ----------- | ------- |
| 1     | Fast        | ~5 ms   |
| 5     | Balanced    | ~10 ms  |
| 100   | High recall | ~20 ms  |
| 1000  | Exhaustive  | ~50 ms  |

---

## 🛠️ Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # add GOOGLE_API_KEY if you want /vector/ask to work
```

### OCR Dependencies

* **Ubuntu/Debian**: `sudo apt install tesseract-ocr`
* **macOS**: `brew install tesseract`
* **Windows**: Install from UB Mannheim build

---

## ▶️ Running the Service

### Docker Compose (Recommended)

Both backend and frontend together:

```bash
docker-compose up
```

Then open:
- **Frontend:** http://localhost:8501
- **Backend Swagger:** http://localhost:8000/docs

### Local Setup (Advanced)

If you prefer to run locally (requires Python 3.10+, Tesseract OCR, and Visual C++ build tools on Windows):

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then in another terminal, run the frontend (see next section).

---

## 🖥️ Streamlit Frontend

A lightweight [Streamlit](https://streamlit.io/) UI ([`streamlit_app.py`](streamlit_app.py)) sits over the API, with a tab for each endpoint:

- **📥 Index** — drag-and-drop PDF/image upload
- **🔍 Search** — semantic search with Hybrid/Rerank toggles and metadata filters
- **💬 Ask** — RAG answers with expandable citations
- **📊 Stats** — index statistics

It talks to the backend purely over HTTP (nothing imported from `app/`), so the API can run anywhere.

### Running with Docker Compose (Recommended)

```bash
docker-compose up
```

Open http://localhost:8501. The frontend automatically connects to the backend at `http://backend:8000` (on the Docker network).

### Running Locally (Advanced)

If running the backend locally:

```bash
python -m venv venv-frontend
venv-frontend\Scripts\activate        # Windows
# source venv-frontend/bin/activate   # macOS/Linux
pip install -r requirements-frontend.txt
streamlit run streamlit_app.py
```

The sidebar lets you change the backend URL (or set `API_BASE_URL`; defaults to `http://localhost:8000`).

---

## 📚 Tech Stack

* FastAPI
* ChromaDB
* Sentence Transformers (dense embeddings + cross-encoder reranking)
* rank-bm25 (sparse/keyword retrieval)
* Google AI Studio / Gemini API (RAG generation, free tier)
* DuckDB + Parquet
* Tesseract OCR
* Pydantic
* Streamlit (frontend UI)

---

## ✅ Summary

This system is designed for:

* Scalable semantic search
* Production-grade reliability
* Configurable performance trade-offs
* Clean separation of concerns

It is suitable for **RAG pipelines, document search, enterprise knowledge bases, and AI assistants**.

## ⚡ Performance & Architecture

This section explains **index type**, **time complexity**, **metadata filtering strategy**, and **memory vs speed trade-offs** used in this system. The goal is to make design decisions explicit and defensible in production reviews.

---

### 1️⃣ Index Type: HNSW (Approximate Nearest Neighbor)

Chroma DB internally uses an **Approximate Nearest Neighbor (ANN)** index based on **HNSW (Hierarchical Navigable Small World graphs)**.

**Why HNSW?**

* Exact KNN requires comparing a query vector with *all* stored vectors → **O(n)** per search (too slow).
* HNSW organizes vectors into a **multi-layer graph**, allowing fast navigation to close neighbors.

**Key Properties:**

* Graph-based, not tree-based
* Multi-layer (coarse → fine)
* High recall (95–99%) with much lower latency

**Result:**

> Near-optimal results with **logarithmic-time search** instead of linear scans.

---

### 2️⃣ Time Complexity

#### Indexing (Ingestion)

| Step                 | Complexity | Explanation                          |
| -------------------- | ---------- | ------------------------------------ |
| OCR extraction       | O(n)       | n = number of image pixels / pages   |
| Text chunking        | O(m)       | m = text length                      |
| Embedding generation | O(c × d)   | c = chunks, d = embedding dims (384) |
| HNSW insertion       | O(log² N)  | N = total stored embeddings          |
| Persistence          | O(c)       | Written to disk (Parquet)            |

**Total (per document):**

```
O(m + c·d + log²N)
```

---

#### Searching

| Step               | Complexity   | Explanation                   |
| ------------------ | ------------ | ----------------------------- |
| Query embedding    | O(d)         | Single vector                 |
| ANN traversal      | O(log N × k) | k = top results               |
| Metadata filtering | O(k)         | Filter applied to ANN results |

**Total (per query):**

```
O(d + logN · k)
```

Typical latency: **5–30 ms** for up to **1M chunks** on CPU.

---

### 3️⃣ Metadata Filtering Strategy

This system uses **post-filtering**, not pre-filtering.

#### Flow

```
Query embedding
   ↓
ANN vector search (semantic relevance)
   ↓
Metadata filtering (user_id, tags, doc_id)
   ↓
Return top_k results
```

#### Why Post-Filtering?

| Strategy      | Pros                 | Cons                        |
| ------------- | -------------------- | --------------------------- |
| Pre-filter    | Smaller search space | Can miss relevant results   |
| Post-filter ✅ | Best semantic recall | Slight extra filtering cost |
| Hybrid        | Best accuracy        | Higher complexity           |

**Reasoning:**

* Semantic relevance is more important than metadata constraints
* Filters usually narrow results after relevance is known
* Filtering cost is negligible (O(k))

---

### 4️⃣ Memory vs Speed Trade-Offs

#### Chunk Size

| Smaller Chunks   | Larger Chunks           |
| ---------------- | ----------------------- |
| Better precision | Fewer embeddings        |
| More embeddings  | Faster indexing         |
| Higher memory    | Lower memory            |
| Better QA        | Worse fine-grain recall |

**Rule of thumb:**

* QA / RAG → 200–400 tokens
* Document search → 500–1000 tokens

---

#### In-Memory vs Persistent Index

| In-Memory         | Persistent (DuckDB + Parquet) |
| ----------------- | ----------------------------- |
| Fastest           | Slight I/O overhead           |
| Lost on restart ❌ | Crash-safe ✅                  |
| RAM-bound         | Disk-scalable                 |

**Current Choice:** Persistent storage for durability with ~5–10% latency cost.

---

### ✅ Summary

* **Index**: HNSW ANN (fast, scalable)
* **Search Complexity**: O(log N)
* **Filtering**: Post-filtering for semantic accuracy
* **Trade-offs**: Memory ↔ Precision ↔ Speed explicitly controlled
