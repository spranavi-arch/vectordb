
from pydantic import BaseModel
from typing import List, Dict, Optional


class IndexResponse(BaseModel):
    """
    Response when successfully indexing a document.
    
    Attributes:
        document_id: Unique identifier for the indexed document
        total_chunks: Total number of text chunks created from the document
    """
    document_id: str
    total_chunks: int


class SearchFilters(BaseModel):
    """
    Optional filters for search queries.
    
    Attributes:
        user_id: Filter by document owner
        document_id: Filter by specific document
        document_name: Filter by document name
        tags: Filter by one or more tags (all tags must be present in document)
        page_number: Filter by page number (if applicable)
    """
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    tags: Optional[List[str]] = None
    page_number: Optional[str] = None


class SearchRequest(BaseModel):
    """
    Request body for semantic vector search.

    The search process:
    1. Query text is converted to vector embedding
    2. Find embeddings closest to query embedding (semantic similarity)
    3. Apply metadata filters if provided
    4. Return top_k most relevant results

    Attributes:
        query: The search query text
        top_k: Maximum number of results to return (default: 5)
        filters: Optional metadata filters to narrow results
        hybrid: Use dense+BM25 hybrid retrieval (Reciprocal Rank Fusion)
            instead of dense-only search
        rerank: Apply cross-encoder reranking to the retrieved candidates
            for higher precision
    """
    query: str
    top_k: int = 5
    filters: Optional[SearchFilters] = None
    hybrid: bool = False
    rerank: bool = False

    class Config:
        # Example data for API documentation
        schema_extra = {
            "example": {
                "query": "",
                "top_k": 5,
                "filters": {

                },
                "hybrid": False,
                "rerank": False
            }
        }


class AskRequest(BaseModel):
    """
    Request body for the RAG /vector/ask endpoint.

    Attributes:
        query: The question to answer
        top_k: Number of context chunks to retrieve (default: 5)
        filters: Optional metadata filters to narrow retrieval
        hybrid: Use dense+BM25 hybrid retrieval for context
        rerank: Apply cross-encoder reranking to the retrieved context
    """
    query: str
    top_k: int = 5
    filters: Optional[SearchFilters] = None
    hybrid: bool = False
    rerank: bool = False


class Citation(BaseModel):
    """
    A single source citation backing part of a generated answer.

    Attributes:
        index: The bracketed number ("[1]") used to reference this source
            inline in the answer text
        document_id: ID of the source document
        document_name: Human-readable name of the source document
        page_number: Page the cited chunk came from
        chunk_text: The retrieved chunk text this citation refers to
        distance: Similarity distance/score for the chunk, if available
    """
    index: int
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    page_number: Optional[int] = None
    chunk_text: str
    distance: Optional[float] = None


class AskResponse(BaseModel):
    """
    Response from the RAG /vector/ask endpoint.

    Attributes:
        answer: Generated answer, citing sources as [1], [2], ...
        citations: Source chunks backing the answer, indexed to match the
            bracketed citation numbers in the answer text
    """
    answer: str
    citations: List[Citation]

