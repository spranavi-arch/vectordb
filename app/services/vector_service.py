
import uuid
import datetime
import logging
from typing import List, Dict, Optional

from rank_bm25 import BM25Okapi

from app.schemas.vector_schemas import SearchFilters
from app.core.exceptions import EmptyQueryError
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.ocr_service import OCRService
from app.services.rerank_service import RerankService
from app.repositories.chroma_repository import ChromaRepository

logger = logging.getLogger(__name__)


class VectorService:
    """
    Orchestrates all vector search operations.
    
    This service is the main business logic layer that coordinates:
    - Document OCR extraction
    - Text chunking with overlap
    - Vector embedding generation
    - Database storage and retrieval
    """
    
    def __init__(
        self,
        chunker: ChunkingService,
        embedder: EmbeddingService,
        ocr: OCRService,
        repo: ChromaRepository,
        reranker: Optional[RerankService] = None,
        rerank_overfetch: int = 4
    ):
        """
        Initialize the vector service with required dependencies.

        Args:
            chunker: Service for splitting text into chunks
            embedder: Service for converting text to embeddings
            ocr: Service for extracting text from documents
            repo: Repository for storing/retrieving embeddings
            reranker: Optional cross-encoder reranking service. Required
                only if callers request rerank=True via retrieve().
            rerank_overfetch: Candidate-pool multiplier used by retrieve()
                when reranking is requested
        """
        self.chunker = chunker
        self.embedder = embedder
        self.ocr = ocr
        self.repo = repo
        self.reranker = reranker
        self.rerank_overfetch = rerank_overfetch

    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """
        Convert metadata dict to database-compatible format.
        
        Chroma stores metadata as strings, so lists are converted to
        comma-separated strings. This preserves all metadata types while
        ensuring database compatibility.
        
        Args:
            metadata: Dictionary with mixed types
            
        Returns:
            Dictionary with all values as strings or basic types
        """
        sanitized = {}
        for k, v in metadata.items():
            if isinstance(v, list):
                # Convert list to comma-separated string (e.g., tags)
                sanitized[k] = ",".join(v)
            else:
                sanitized[k] = v
        return sanitized

    def index_document(
        self,
        file_bytes: bytes,
        content_type: str,
        user_id: str,
        tags: List[str],
        document_name: str 
    ):
        """
        Index a document by converting it to embeddings and storing in database.
        
        Complete workflow:
        1. Generate unique document ID
        2. Extract text from document using OCR (handles PDFs & images)
        3. For each page:
           a. Split text into overlapping chunks
           b. Generate embeddings for each chunk
           c. Store embeddings with metadata
        
        Args:
            file_bytes: Raw file content (PDF or image)
            content_type: MIME type (application/pdf or image/*)
            user_id: Owner of the document
            tags: List of tags for categorization
            document_name: Human-readable name for the document
            
        Returns:
            Tuple of (document_id, total_chunks_created)
            
        Example:
            doc_id, chunks = service.index_document(
                file_bytes=pdf_data,
                content_type="application/pdf",
                user_id="user123",
                tags=["medical", "report"],
                document_name="Lab Results 2024"
            )
        """
        # Generate unique ID for this document
        document_id = str(uuid.uuid4())

        # Extract text from PDF or image
        ocr_pages = self.ocr.extract_text(file_bytes, content_type)

        # Determine actual source type the same way OCR did (sniffed from
        # file bytes, not the possibly-inaccurate content_type header) so
        # stored metadata stays consistent with what was actually parsed.
        source_type = "pdf" if self.ocr.is_pdf(file_bytes, content_type) else "image"

        total_chunks = 0

        # Process each page separately
        for page_number, page_text in ocr_pages:
            # Split page text into overlapping chunks
            chunks = self.chunker.chunk_text(page_text)

            # Skip blank/empty pages: nothing to embed or store
            if not chunks:
                continue

            # Convert all chunks to embeddings in one batch
            embeddings = self.embedder.embed(chunks)

            ids = []
            metadatas = []

            # Prepare data for database storage
            for i, chunk in enumerate(chunks):
                # Generate unique ID for this chunk
                ids.append(str(uuid.uuid4()))

                # Create metadata for this chunk
                metadata = {
                    "source": source_type,
                    "page_number": page_number,
                    "chunk_index": i,  # Order within page
                    "created_at": datetime.datetime.utcnow().isoformat(),
                    "tags": tags,
                    "document_id": document_id,
                    "document_name": document_name,
                    "user_id": user_id,
                }

                # Convert metadata to database-compatible format
                metadatas.append(self._sanitize_metadata(metadata))

            # Store all chunks from this page
            self.repo.add(ids, embeddings, chunks, metadatas)
            total_chunks += len(chunks)

        logger.info(f"Indexed document {document_id} with {total_chunks} chunks")

        return document_id, total_chunks

    def search(
        self,
        query: str,
        filters,
        top_k: int
    ):
        """
        Search for documents similar to the query using vector similarity.
        
        Process:
        1. Validate query is not empty
        2. Validate top_k is positive
        3. Convert query to embedding
        4. Search database for most similar embeddings
        5. Post-filter results by tags in Python
        6. Return top_k results with scores
        
        Args:
            query: Search query text
            filters: Optional SearchFilters object for metadata filtering
            top_k: Maximum number of results to return
            
        Returns:
            Search results with embeddings, distances, and metadata
            
        Raises:
            ValueError: If query is empty or top_k invalid
            
        Example:
            results = service.search(
                query="What are the symptoms?",
                filters=SearchFilters(user_id="user123", tags=["medical"]),
                top_k=5
            )
        """
        # Validate query
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        
        # Validate top_k
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {top_k}")

        # Convert query text to embedding
        query_embedding = self.embedder.embed(query)

        # Build database filter clause from user-provided filters
        # Note: Tag filters are handled separately via post-filtering
        where_clause = self._build_where_clause(filters)

        # Post-filter results by tags if requested
        if filters and filters.tags:
            results = self._search_with_tag_filter(
                query_embedding=query_embedding,
                where_clause=where_clause,
                tags=filters.tags,
                top_k=top_k
            )
        else:
            # Search database for similar embeddings
            # Note: We fetch top_k results (no over-fetching needed without tag filters)
            results = self.repo.search(
                query_embedding=query_embedding,
                filters=where_clause,
                top_k=top_k
            )

        return results

    def hybrid_search(self, query: str, filters, top_k: int):
        """
        Search using both dense (embedding) and sparse (BM25 keyword)
        retrieval, fused via Reciprocal Rank Fusion (RRF). Complements pure
        semantic search with exact keyword matching, which embeddings can
        miss (IDs, codes, rare proper nouns, exact phrases).

        Trade-off: the BM25 index is rebuilt in-memory from the (optionally
        filtered) candidate pool on every call - O(pool size) per query.
        Fine at portfolio scale; a production system would maintain a
        persistent/incremental sparse index (e.g. Elasticsearch, Typesense)
        instead of rebuilding one per request.

        Args:
            query: Search query text
            filters: Optional SearchFilters object for metadata filtering
            top_k: Maximum number of fused results to return

        Returns:
            Results in the same shape as search() - a drop-in alternative
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {top_k}")

        where_clause = self._build_where_clause(filters)
        # Candidate pool size per ranking before fusion - wider than top_k so
        # fusion has enough signal to work with from both rankings.
        fetch_k = max(top_k * 4, 20)

        # --- Dense ranking ---
        dense_results = self.repo.search(
            query_embedding=self.embedder.embed(query),
            filters=where_clause,
            top_k=fetch_k
        )
        dense_ids = dense_results["ids"][0] if dense_results.get("ids") else []
        dense_documents = dense_results["documents"][0] if dense_results.get("documents") else []
        dense_metadatas = dense_results["metadatas"][0] if dense_results.get("metadatas") else []
        dense_distances = dense_results["distances"][0] if dense_results.get("distances") else []
        dense_index_by_id = {doc_id: i for i, doc_id in enumerate(dense_ids)}

        # --- Sparse (BM25) ranking over the same filtered candidate pool ---
        pool = self.repo.get_all(filters=where_clause)
        pool_ids = pool.get("ids") or []
        pool_documents = pool.get("documents") or []
        pool_metadatas = pool.get("metadatas") or []
        id_to_document = dict(zip(pool_ids, pool_documents))
        id_to_metadata = dict(zip(pool_ids, pool_metadatas))

        sparse_ids = []
        if pool_ids:
            tokenized_corpus = [doc.lower().split() for doc in pool_documents]
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(query.lower().split())
            ranked = sorted(zip(pool_ids, scores), key=lambda pair: pair[1], reverse=True)
            # Drop zero-score matches - a doc with no query-term overlap at
            # all shouldn't outrank a lower-fused-score doc just because it
            # happened to appear in the pool.
            sparse_ids = [doc_id for doc_id, score in ranked[:fetch_k] if score > 0]

        # --- Fuse the two rankings ---
        fused_scores = self._reciprocal_rank_fusion([dense_ids, sparse_ids])
        fused_order = sorted(fused_scores, key=lambda doc_id: fused_scores[doc_id], reverse=True)

        fused_ids, fused_documents, fused_metadatas, fused_distances = [], [], [], []
        for doc_id in fused_order:
            if doc_id in dense_index_by_id:
                idx = dense_index_by_id[doc_id]
                document, metadata, distance = dense_documents[idx], dense_metadatas[idx], dense_distances[idx]
            else:
                # Keyword-only match: no embedding distance to report
                document = id_to_document.get(doc_id, "")
                metadata = id_to_metadata.get(doc_id, {})
                distance = None

            fused_ids.append(doc_id)
            fused_documents.append(document)
            fused_metadatas.append(metadata)
            fused_distances.append(distance)

        results = {
            "ids": [fused_ids],
            "documents": [fused_documents],
            "metadatas": [fused_metadatas],
            "distances": [fused_distances],
        }

        if filters and filters.tags:
            results = self._filter_results_by_tags(results, filters.tags)

        results["ids"] = [results["ids"][0][:top_k]]
        results["documents"] = [results["documents"][0][:top_k]]
        results["metadatas"] = [results["metadatas"][0][:top_k]]
        results["distances"] = [results["distances"][0][:top_k]]

        return results

    @staticmethod
    def _reciprocal_rank_fusion(rankings: List[List[str]], k: int = 60) -> Dict[str, float]:
        """
        Fuse multiple ranked ID lists into one score per ID via Reciprocal
        Rank Fusion: score(d) = sum over rankings containing d of
        1/(k + rank). k=60 is the standard RRF constant from the original
        paper - it dampens the impact of a document's exact rank so results
        from rankings on very different score scales (cosine distance vs.
        BM25 score) combine sensibly without needing to tune a weighted
        linear blend.
        """
        scores: Dict[str, float] = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return scores

    def retrieve(self, query: str, filters, top_k: int, hybrid: bool = False, rerank: bool = False):
        """
        Unified retrieval entry point composing dense/hybrid search with
        optional cross-encoder reranking. Used by /vector/search (when the
        hybrid or rerank flags are set) and by AnswerService for RAG
        context, so both endpoints share one retrieval stack rather than
        duplicating retrieval logic.

        When rerank=True, over-fetches rerank_overfetch * top_k candidates
        first - reranking a pool already trimmed to top_k could only
        re-order it, not surface better results that were ranked just
        outside the cutoff by the first-pass retrieval.
        """
        if rerank and not self.reranker:
            raise RuntimeError("Reranking requested but no RerankService is configured")

        fetch_k = top_k * self.rerank_overfetch if rerank else top_k

        results = self.hybrid_search(query, filters, fetch_k) if hybrid else self.search(query, filters, fetch_k)

        if rerank:
            results = self.reranker.rerank(query, results, top_k)

        return results

    def _search_with_tag_filter(self, query_embedding, where_clause, tags, top_k):
        """
        Search and post-filter by tags, progressively widening the fetch
        until top_k matches are found or the whole collection has been
        searched. A fixed over-fetch multiplier can silently under-return
        results when a tag is rare relative to the candidate set.
        """
        total_available = self.repo.count()
        fetch_k = top_k * 4

        while True:
            capped_fetch_k = min(fetch_k, total_available) if total_available else fetch_k

            results = self.repo.search(
                query_embedding=query_embedding,
                filters=where_clause,
                top_k=capped_fetch_k
            )
            filtered = self._filter_results_by_tags(results, tags)

            enough_matches = len(filtered["ids"][0]) >= top_k
            exhausted_collection = total_available and capped_fetch_k >= total_available
            no_more_candidates = not results["ids"] or not results["ids"][0] or \
                len(results["ids"][0]) < capped_fetch_k

            if enough_matches or exhausted_collection or no_more_candidates:
                break

            fetch_k *= 4

        # Trim to requested top_k after filtering
        if filtered["ids"] and filtered["ids"][0]:
            filtered["ids"] = [filtered["ids"][0][:top_k]]
            filtered["distances"] = [filtered["distances"][0][:top_k]]
            filtered["metadatas"] = [filtered["metadatas"][0][:top_k]]
            filtered["documents"] = [filtered["documents"][0][:top_k]]

        return filtered

    def _filter_results_by_tags(self, results, requested_tags):
        """
        Filter search results to only include items with requested tags.
        
        Since tags are stored as comma-separated strings in metadata,
        we need to check if any requested tag appears in the stored tags string.
        
        Args:
            results: Chroma query results
            requested_tags: List of tags to filter by
            
        Returns:
            Filtered results containing only items with matching tags
        """
        filtered_ids = []
        filtered_distances = []
        filtered_metadatas = []
        filtered_documents = []
        
        for i, metadata in enumerate(results["metadatas"][0]):
            stored_tags_str = metadata.get("tags", "")
            stored_tags = [tag.strip() for tag in stored_tags_str.split(",") if tag.strip()]
            
            # Require all requested tags to be present (AND semantics, per
            # SearchFilters.tags contract)
            if all(tag in stored_tags for tag in requested_tags):
                filtered_ids.append(results["ids"][0][i])
                filtered_distances.append(results["distances"][0][i])
                filtered_metadatas.append(metadata)
                filtered_documents.append(results["documents"][0][i])
        
        return {
            "ids": [filtered_ids],
            "distances": [filtered_distances],
            "metadatas": [filtered_metadatas],
            "documents": [filtered_documents]
        }

    def _build_where_clause(
        self,
        filters: Optional[SearchFilters]
    ) -> Optional[Dict]:
        """
        Convert user-friendly SearchFilters to Chroma database where clause.
        
        Chroma where clause syntax:
        - Simple match: {"field": {"$eq": value}}
        - Multiple conditions: {"$and": [condition1, condition2]}
        
        Note: Tag filtering is done via post-filtering in search() method
        since tags are stored as comma-separated strings.
        
        Args:
            filters: Optional SearchFilters object with user_id, tags, etc.
            
        Returns:
            Chroma-compatible where clause dict, or None if no filters
            
        Example:
            Input: SearchFilters(user_id="123", tags=["urgent"])
            Output: {"user_id": {"$eq": "123"}}
            (tags are filtered separately in Python)
        """
        if not filters:
            return None

        conditions = []

        # Filter by owner
        if filters.user_id:
            conditions.append({"user_id": {"$eq": filters.user_id}})

        # Filter by document name
        if filters.document_name:
            conditions.append({"document_name": {"$eq": filters.document_name}})

        # Filter by document id
        if filters.document_id:
            conditions.append({"document_id": {"$eq": filters.document_id}})

        # Filter by page number (stored as int metadata; cast the incoming
        # string filter to match)
        if filters.page_number:
            try:
                page_number = int(filters.page_number)
            except (TypeError, ValueError):
                raise ValueError(
                    f"page_number filter must be an integer, got: {filters.page_number}"
                )
            conditions.append({"page_number": {"$eq": page_number}})

        # Note: Tag filtering is handled in _filter_results_by_tags() method
        # via post-filtering, since tags are stored as comma-separated strings

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}
