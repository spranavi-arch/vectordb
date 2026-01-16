
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
    """
    query: str
    top_k: int = 5
    filters: Optional[SearchFilters] = None

    class Config:
        # Example data for API documentation
        schema_extra = {
            "example": {
                "query": "",
                "top_k": 5,
                "filters": {
                    
                }
            }
        }
 





