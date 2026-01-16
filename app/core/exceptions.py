"""
Custom exception classes for the Vector Search Service.

These exceptions represent specific error conditions that can occur during
vector search operations. They inherit from a base VectorServiceError for
easy exception grouping and handling.
"""


class VectorServiceError(Exception):
    """Base exception for all Vector Search Service errors."""
    http_status_code = 500


class EmptyQueryError(VectorServiceError):
    """Raised when a search query is empty or contains only whitespace."""
    http_status_code = 400


class InvalidFilterError(VectorServiceError):
    """Raised when search filters are invalid or malformed."""
    http_status_code = 400


class DimensionMismatchError(VectorServiceError):
    """Raised when embedding dimensions don't match expected size."""
    http_status_code = 409


class MissingEmbeddingsError(VectorServiceError):
    """Raised when embeddings are missing or empty."""
    http_status_code = 404


class VectorIndexNotInitializedError(VectorServiceError):
    """Raised when attempting to search before the vector database is initialized."""
    http_status_code = 503


class EmptyIndexError(VectorServiceError):
    """Raised when index has no embeddings to search."""
    http_status_code = 404


class DatabaseError(VectorServiceError):
    """Raised when database operations fail."""
    http_status_code = 500
