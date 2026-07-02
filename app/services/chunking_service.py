
import re

def split_paragraphs(text: str) -> list[str]:
    """
    Split text into paragraphs by detecting blank lines.
    Preserves semantic boundaries for better chunk coherence.
    
    Args:
        text: Raw text to split
        
    Returns:
        List of paragraphs with whitespace stripped
    """
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def estimate_tokens(text: str) -> int:
    return len(text) // 4


class ChunkingService:
    def __init__(self, max_tokens: int = 300, overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_text(self, text: str) -> list[str]:
        paragraphs = split_paragraphs(text)
        chunks = []
        current = []

        current_tokens = 0

        for para in paragraphs:
            tokens = estimate_tokens(para)

            if current_tokens + tokens > self.max_tokens:
                if current:
                    chunks.append(" ".join(current))

                    # overlap last part
                    overlap_text = " ".join(current)[-self.overlap_tokens * 4:]
                    current = [overlap_text]
                    current_tokens = estimate_tokens(overlap_text)
                else:
                    current = []
                    current_tokens = 0

                if tokens > self.max_tokens:
                    # Single paragraph alone exceeds max_tokens: hard-split it
                    # by character length so no chunk ever exceeds the bound.
                    for piece in self._split_oversized(para):
                        chunks.append(piece)
                else:
                    current.append(para)
                    current_tokens += tokens
            else:
                current.append(para)
                current_tokens += tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _split_oversized(self, text: str) -> list[str]:
        """Hard-split a single paragraph that alone exceeds max_tokens into
        character-sliced pieces (with overlap) so no chunk exceeds the bound."""
        max_chars = self.max_tokens * 4
        overlap_chars = self.overlap_tokens * 4

        pieces = []
        start = 0
        while start < len(text):
            end = start + max_chars
            piece = text[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= len(text):
                break
            start = end - overlap_chars

        return pieces
