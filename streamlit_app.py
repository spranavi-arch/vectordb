"""
Streamlit frontend for the Vector Search Service.

A thin UI over the FastAPI backend (app.main). It talks to the four
endpoints over HTTP — nothing is imported from the backend directly, so the
API can run anywhere (locally, Docker, remote) as long as the base URL points
at it.

Tabs:
  - Index   → POST /vector/index   (multipart file upload)
  - Search  → POST /vector/search  (semantic / hybrid / reranked)
  - Ask     → POST /vector/ask     (RAG answer with citations)
  - Stats   → GET  /vector/stats   (+ /health)

Run with:
    streamlit run streamlit_app.py

The backend must already be running (uvicorn app.main:app --reload).
"""

import os
import requests
import streamlit as st

# Default to the local uvicorn address; override with API_BASE_URL if the
# service runs elsewhere (Docker, remote host, different port).
DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Vector Search Service", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------------------
# Sidebar: connection settings + live health check
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🔎 Vector Search")
    api_base = st.text_input("API base URL", value=DEFAULT_API_BASE).rstrip("/")

    st.caption("Backend must be running: `uvicorn app.main:app --reload`")

    # Ping /health so the user immediately sees whether the backend is up
    # and how many chunks are indexed.
    try:
        health = requests.get(f"{api_base}/health", timeout=5).json()
        st.success(f"Connected — {health.get('indexed_chunks', 0)} chunks indexed")
    except Exception:
        st.error("Cannot reach backend. Check the URL and that it's running.")


def build_filters(user_id, document_id, document_name, tags_str):
    """Assemble a SearchFilters dict, omitting empty fields.

    The backend treats an all-empty filter object the same as no filter, but
    we drop empties anyway to keep requests clean and avoid filtering on "".
    """
    filters = {}
    if user_id.strip():
        filters["user_id"] = user_id.strip()
    if document_id.strip():
        filters["document_id"] = document_id.strip()
    if document_name.strip():
        filters["document_name"] = document_name.strip()
    tags = [t.strip() for t in tags_str.split(",") if t.strip()]
    if tags:
        filters["tags"] = tags
    return filters or None


tab_index, tab_search, tab_ask, tab_stats = st.tabs(
    ["📥 Index", "🔍 Search", "💬 Ask (RAG)", "📊 Stats"]
)


# ---------------------------------------------------------------------------
# Index tab → POST /vector/index
# ---------------------------------------------------------------------------
with tab_index:
    st.header("Index a document")
    st.caption("Upload a PDF or image. Text is OCR'd, chunked, embedded, and stored.")

    uploaded = st.file_uploader(
        "Document (PDF or image)",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
    )
    col1, col2 = st.columns(2)
    with col1:
        idx_user_id = st.text_input("User ID", value="user123", key="idx_user")
    with col2:
        idx_doc_name = st.text_input("Document name (optional)", key="idx_name")
    idx_tags = st.text_input(
        "Tags (comma-separated)", value="", placeholder="medical, urgent", key="idx_tags"
    )

    if st.button("Index document", type="primary", disabled=uploaded is None):
        if not idx_user_id.strip():
            st.error("User ID is required.")
        elif not idx_tags.strip():
            # The backend requires the tags form field (Form(...)), so guard here.
            st.error("At least one tag is required.")
        else:
            with st.spinner("Indexing… (OCR + embedding can take a moment)"):
                try:
                    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                    data = {"user_id": idx_user_id.strip(), "tags": idx_tags.strip()}
                    if idx_doc_name.strip():
                        data["document_name"] = idx_doc_name.strip()
                    resp = requests.post(
                        f"{api_base}/vector/index", files=files, data=data, timeout=300
                    )
                    if resp.ok:
                        body = resp.json()
                        st.success(
                            f"Indexed **{uploaded.name}** — "
                            f"{body['total_chunks']} chunks created."
                        )
                        st.code(body["document_id"], language=None)
                        st.caption("↑ document_id — use it as a Search/Ask filter.")
                    else:
                        st.error(f"{resp.status_code}: {resp.json().get('detail', resp.text)}")
                except Exception as e:
                    st.error(f"Request failed: {e}")


# ---------------------------------------------------------------------------
# Shared retrieval controls (Search + Ask both use these)
# ---------------------------------------------------------------------------
def retrieval_controls(key_prefix):
    """Render top_k / hybrid / rerank + filter inputs; return their values."""
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        top_k = st.slider("top_k", 1, 50, 5, key=f"{key_prefix}_topk")
    with c2:
        hybrid = st.toggle("Hybrid (BM25)", key=f"{key_prefix}_hybrid")
    with c3:
        rerank = st.toggle("Rerank", key=f"{key_prefix}_rerank")

    with st.expander("Filters (optional)"):
        f1, f2 = st.columns(2)
        with f1:
            user_id = st.text_input("user_id", key=f"{key_prefix}_f_user")
            document_id = st.text_input("document_id", key=f"{key_prefix}_f_docid")
        with f2:
            document_name = st.text_input("document_name", key=f"{key_prefix}_f_docname")
            tags = st.text_input("tags (comma-separated)", key=f"{key_prefix}_f_tags")
    filters = build_filters(user_id, document_id, document_name, tags)
    return top_k, hybrid, rerank, filters


# ---------------------------------------------------------------------------
# Search tab → POST /vector/search
# ---------------------------------------------------------------------------
with tab_search:
    st.header("Semantic search")
    query = st.text_input(
        "Query", placeholder="What are the symptoms of hypertension?", key="search_query"
    )
    top_k, hybrid, rerank, filters = retrieval_controls("search")

    if st.button("Search", type="primary", key="search_btn"):
        if not query.strip():
            st.error("Query cannot be empty.")
        else:
            payload = {
                "query": query,
                "top_k": top_k,
                "hybrid": hybrid,
                "rerank": rerank,
                "filters": filters,
            }
            with st.spinner("Searching…"):
                try:
                    resp = requests.post(
                        f"{api_base}/vector/search", json=payload, timeout=120
                    )
                    if not resp.ok:
                        st.error(f"{resp.status_code}: {resp.json().get('detail', resp.text)}")
                    else:
                        results = resp.json()
                        # Chroma returns nested lists ([[...]] for one query).
                        ids = (results.get("ids") or [[]])[0]
                        docs = (results.get("documents") or [[]])[0]
                        dists = (results.get("distances") or [[]])[0]
                        metas = (results.get("metadatas") or [[]])[0]
                        scores = (results.get("rerank_scores") or [[]])
                        scores = scores[0] if scores and isinstance(scores[0], list) else scores

                        if not ids:
                            st.info(results.get("message", "No matching documents found."))
                        else:
                            st.caption(f"{len(ids)} results")
                            for i, chunk_id in enumerate(ids):
                                meta = metas[i] if i < len(metas) else {}
                                name = meta.get("document_name", "?")
                                page = meta.get("page_number", "?")
                                dist = dists[i] if i < len(dists) else None
                                header = f"**{i + 1}. {name}** · page {page}"
                                if dist is not None:
                                    header += f" · distance {dist:.4f}"
                                if rerank and i < len(scores):
                                    header += f" · rerank {scores[i]:.4f}"
                                with st.container(border=True):
                                    st.markdown(header)
                                    st.write(docs[i] if i < len(docs) else "")
                                    with st.expander("metadata"):
                                        st.json(meta)
                except Exception as e:
                    st.error(f"Request failed: {e}")


# ---------------------------------------------------------------------------
# Ask tab → POST /vector/ask  (RAG)
# ---------------------------------------------------------------------------
with tab_ask:
    st.header("Ask (retrieval-augmented generation)")
    st.caption("Retrieves context and asks Gemini to answer with inline citations. "
               "Requires GOOGLE_API_KEY in the backend's .env.")
    ask_query = st.text_input(
        "Question", placeholder="What are the symptoms of hypertension?", key="ask_query"
    )
    a_top_k, a_hybrid, a_rerank, a_filters = retrieval_controls("ask")

    if st.button("Ask", type="primary", key="ask_btn"):
        if not ask_query.strip():
            st.error("Question cannot be empty.")
        else:
            payload = {
                "query": ask_query,
                "top_k": a_top_k,
                "hybrid": a_hybrid,
                "rerank": a_rerank,
                "filters": a_filters,
            }
            with st.spinner("Retrieving context and generating answer…"):
                try:
                    resp = requests.post(
                        f"{api_base}/vector/ask", json=payload, timeout=180
                    )
                    if not resp.ok:
                        detail = resp.json().get("detail", resp.text)
                        if resp.status_code == 503:
                            st.warning(f"RAG unavailable: {detail}")
                        else:
                            st.error(f"{resp.status_code}: {detail}")
                    else:
                        body = resp.json()
                        st.markdown("### Answer")
                        st.markdown(body["answer"])
                        citations = body.get("citations", [])
                        if citations:
                            st.markdown("### Citations")
                            for c in citations:
                                label = (
                                    f"[{c['index']}] {c.get('document_name', '?')}"
                                    f" · page {c.get('page_number', '?')}"
                                )
                                if c.get("distance") is not None:
                                    label += f" · distance {c['distance']:.4f}"
                                with st.expander(label):
                                    st.write(c.get("chunk_text", ""))
                except Exception as e:
                    st.error(f"Request failed: {e}")


# ---------------------------------------------------------------------------
# Stats tab → GET /vector/stats
# ---------------------------------------------------------------------------
with tab_stats:
    st.header("Index statistics")
    if st.button("Refresh stats", key="stats_btn"):
        pass  # button just triggers a rerun; the fetch below runs every render
    try:
        stats = requests.get(f"{api_base}/vector/stats", timeout=10).json()
        st.metric("Total chunks indexed", stats.get("total_chunks", 0))
    except Exception as e:
        st.error(f"Could not fetch stats: {e}")
