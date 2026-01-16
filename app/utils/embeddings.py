"""
Utility module for text embedding operations.

Provides helper class for generating text embeddings using pre-trained
sentence transformer models. This is an alternative way to generate embeddings
compared to the EmbeddingService class.
"""

from sentence_transformers import SentenceTransformer
import logging

# Create logger for this module
logger = logging.getLogger("embeddings")


class EmbeddingGenerator:
    """
    Utility class for generating text embeddings.
    
    This is an alternative implementation to EmbeddingService that can be used
    for standalone embedding generation tasks.
    """
    
    def __init__(self):
        """
        Initialize the embedding generator with pre-trained model.
        
        Raises:
            RuntimeError: If the embedding model fails to load
        """
        try:
            # Load the sentence transformer model
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.exception("Embedding model load failed")
            raise RuntimeError("Embedding model failed to load")

    def generate(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (384 dimensions each)
            
        Raises:
            ValueError: If texts list is empty
            
        Example:
            generator = EmbeddingGenerator()
            embeddings = generator.generate([
                "First document",
                "Second document"
            ])
        """
        if not texts:
            raise ValueError("No texts provided for embedding")

        # Encode texts to embeddings and convert to list format
        return self.model.encode(texts).tolist()
