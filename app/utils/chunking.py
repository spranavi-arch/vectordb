"""
Utility functions for text chunking.

Provides helper functions for splitting text into overlapping chunks.
This is a utility module separate from the ChunkingService class.
"""

from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> List[str]:
    """
    Split text into overlapping chunks with customizable size and overlap.
    
    This standalone function provides the same functionality as ChunkingService
    but can be used independently for simple chunking operations.
    
    Args:
        text: The text to chunk
        chunk_size: Size of each chunk in characters (default: 500)
        overlap: Overlap between consecutive chunks (default: 100)
        
    Returns:
        List of text chunks
        
    Raises:
        ValueError: If text is empty or only whitespace
        
    Example:
        chunks = chunk_text(
            "This is a long document text...",
            chunk_size=500,
            overlap=100
        )
    """
    if not text.strip():
        raise ValueError("Text is empty")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks
