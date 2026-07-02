"""
Vector database repository using Chroma DB.

This module handles all interactions with the Chroma vector database,
including adding embeddings, searching, and retrieving statistics.

Chroma is configured with:
- DuckDB+Parquet backend for efficient storage and querying
- Persistent storage to disk for durability
- Automatic indexing for fast similarity searches
"""

import chromadb
from chromadb.config import Settings
import logging

logger = logging.getLogger(__name__)


class ChromaRepository:
    """
    Repository for vector operations using Chroma DB backend.
    
    Handles:
    - Vector storage and persistence
    - Similarity search queries
    - Metadata filtering
    - Database statistics
    """
    
    def __init__(self, persist_dir: str):
        """
        Initialize Chroma repository with persistent storage.
        
        Args:
            persist_dir: Directory path for storing Chroma data files
        """
        self.persist_dir = persist_dir

        # Initialize Chroma client with persistent storage configuration
        self.client = chromadb.Client(
            Settings(
                persist_directory=self.persist_dir,
                # Use DuckDB+Parquet for efficient storage and querying
                chroma_db_impl="duckdb+parquet",
                # Disable telemetry for privacy
                anonymized_telemetry=False
            )
        )

        # Get or create collection for storing document chunks
        # Collection = table for embeddings with metadata
        self.collection = self.client.get_or_create_collection(
            name="document_chunks"
        )

        logger.info(
            f"ChromaDB initialized with duckdb+parquet at {self.persist_dir}"
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ):
        """
        Add embeddings and metadata to the vector database.
        
        Args:
            ids: Unique identifiers for each embedding (UUIDs)
            embeddings: Vector embeddings (list of float lists)
            documents: Original text chunks corresponding to embeddings
            metadatas: Metadata for each embedding (user_id, tags, page_number, etc.)
        """
        # Add data to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        # Force persist data to disk to ensure durability
        # Without this, data might stay in memory only
        self.client.persist()
        logger.info(f"Persisted {len(ids)} vectors to disk")

    def search(self, query_embedding, filters, top_k: int):
        """
        Search for similar embeddings using vector similarity.
        
        Uses cosine similarity to find embeddings closest to the query embedding.
        Optionally filters results by metadata (user_id, tags, etc.).
        
        Args:
            query_embedding: The query vector (float list)
            filters: Optional metadata filters (Chroma where clause)
            top_k: Number of results to return
            
        Returns:
            Query results from Chroma with IDs, distances, and metadata
        """
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )

    def get_all(self, filters=None):
        """
        Fetch all documents (with metadata) matching an optional where clause,
        without a similarity query. Used to build the candidate pool for
        keyword-based (BM25) scoring in hybrid search.

        Args:
            filters: Optional Chroma where clause to narrow the pool

        Returns:
            dict with "ids", "documents", "metadatas" (flat lists, not
            nested like query() results)
        """
        return self.collection.get(where=filters, include=["documents", "metadatas"])

    def count(self) -> int:
        """
        Get total number of chunks stored in the database.
        
        Returns:
            Total count of embeddings in the collection
        """
        return self.collection.count()
    
    def stats(self):
        """
        Get database statistics.
        
        Returns:
            dict with statistics like total_chunks
        """
        return {
            "total_chunks": self.collection.count()
        }
