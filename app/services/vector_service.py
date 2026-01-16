
import uuid
import datetime
import logging
from typing import List, Dict, Optional

from app.schemas.vector_schemas import SearchFilters
from app.core.exceptions import EmptyQueryError
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.ocr_service import OCRService
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
        repo: ChromaRepository
    ):
        """
        Initialize the vector service with required dependencies.
        
        Args:
            chunker: Service for splitting text into chunks
            embedder: Service for converting text to embeddings
            ocr: Service for extracting text from documents
            repo: Repository for storing/retrieving embeddings
        """
        self.chunker = chunker
        self.embedder = embedder
        self.ocr = ocr
        self.repo = repo

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

        total_chunks = 0

        # Process each page separately
        for page_number, page_text in ocr_pages:
            # Split page text into overlapping chunks
            chunks = self.chunker.chunk_text(page_text)
            
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
                    "source": "pdf" if content_type == "application/pdf" else "image",
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

        # Search database for similar embeddings
        # Note: We fetch top_k results (no over-fetching needed without tag filters)
        results = self.repo.search(
            query_embedding=query_embedding,
            filters=where_clause,
            top_k=top_k
        )
        
        # Post-filter results by tags if requested
        if filters and filters.tags:
            # Retrieve extra results to account for tag filtering reducing results
            results = self.repo.search(
                query_embedding=query_embedding,
                filters=where_clause,
                top_k=top_k * 2  # Get extra results for tag filtering
            )
            results = self._filter_results_by_tags(results, filters.tags)
            # Trim to requested top_k after filtering
            if results["ids"] and results["ids"][0]:
                results["ids"] = [results["ids"][0][:top_k]]
                results["distances"] = [results["distances"][0][:top_k]]
                results["metadatas"] = [results["metadatas"][0][:top_k]]
                results["documents"] = [results["documents"][0][:top_k]]
        
        return results

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
            
            # Check if any requested tag is in the stored tags
            if any(tag in stored_tags for tag in requested_tags):
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

        # Note: Tag filtering is handled in _filter_results_by_tags() method
        # via post-filtering, since tags are stored as comma-separated strings

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}
