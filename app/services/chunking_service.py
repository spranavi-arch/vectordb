
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
                chunks.append(" ".join(current))

                # overlap last part
                overlap_text = " ".join(current)[-self.overlap_tokens * 4:]
                current = [overlap_text, para]
                current_tokens = estimate_tokens(overlap_text) + tokens
            else:
                current.append(para)
                current_tokens += tokens

        if current:
            chunks.append(" ".join(current))

        return chunks
