
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for converting text strings to vector embeddings.
    
    Uses sentence-transformers pre-trained models that generate semantically
    meaningful embeddings suitable for similarity searches.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the embedding service with a pre-trained model.
        
        Args:
            model_name: HuggingFace sentence-transformers model name
                       (e.g., "sentence-transformers/all-MiniLM-L6-v2")
                       
        Raises:
            Exception: If model fails to load
        """
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of text strings to vector embeddings.
        
        Each string is encoded into a 384-dimensional vector where:
        - Similar texts have vectors pointing in similar directions
        - Distant texts have vectors pointing in different directions
        - This enables cosine similarity search
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (list of float lists)
            Each vector has 384 dimensions
        """
        # encode: Convert texts to embeddings
        # normalize_embeddings: Normalize to unit vectors for cosine similarity
        # tolist(): Convert numpy arrays to Python lists
        return self.model.encode(texts, normalize_embeddings=True).tolist()
