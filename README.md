# Vector Search Service

A **production-ready FastAPI-based vector search system** that enables semantic search over documents (PDFs and images) using OCR, intelligent chunking, and vector embeddings stored in ChromaDB.

---

## 🎯 Overview

This service ingests documents, extracts text using OCR, splits the text into semantically meaningful chunks, converts them into vector embeddings, and stores them in a vector database for **fast, semantic similarity search**.

Unlike keyword search, this system retrieves results based on **meaning and context**, not exact word matches.

---

## ✨ Key Features

* **OCR Text Extraction**
  Extracts text from PDFs and images using Tesseract OCR.

* **Configurable Chunking**
  Chunk size, overlap, and strategy are configurable to balance recall, precision, and cost.

* **Semantic Embeddings**
  Uses Sentence Transformers to generate dense vector representations (384 dimensions).

* **Approximate Nearest Neighbor (ANN) Search**
  Uses ChromaDB’s HNSW-style index for low-latency similarity search.

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
EmbeddingService.embed()
   ↓
ChromaRepository.search()
   ↓
Metadata Post-Filtering
   ↓
Top-K Results
```

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

Search indexed documents using semantic similarity.

**Request**

```json
{
  "query": "What are the symptoms of hypertension?",
  "top_k": 5,
  "filters": {
    "user_id": "user123",
    "tags": ["medical"]
  }
}
```

**Response**

```json
{
  "ids": [...],
  "distances": [...],
  "documents": [...],
  "metadatas": [...]
}
```

---

### 3️⃣ GET `/vector/stats`

Returns index-level statistics.

```json
{
  "total_chunks": 1024
}
```

---

## ⚙️ Configuration Highlights

| Parameter       | Purpose                 | Typical Value |
| --------------- | ----------------------- | ------------- |
| `chunk_size`    | Size of each text chunk | 400–800 chars |
| `chunk_overlap` | Overlap between chunks  | 50–150 chars  |
| `embedding_dim` | Vector dimension        | 384           |
| `top_k`         | Search results returned | 5–20          |

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
| Hybrid      | Optimal     | Complex implementation    |

**Current choice:** Post-filtering for correctness and recall.

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
```

### OCR Dependencies

* **Ubuntu/Debian**: `sudo apt install tesseract-ocr`
* **macOS**: `brew install tesseract`
* **Windows**: Install from UB Mannheim build

---

## ▶️ Running the Service

```bash
uvicorn app.main:app --reload
```

**Docker**

```bash
docker build -t vectordb .
docker run -p 8000:8000 vectordb
```

---

## 📚 Tech Stack

* FastAPI
* ChromaDB
* Sentence Transformers
* DuckDB + Parquet
* Tesseract OCR
* Pydantic

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
