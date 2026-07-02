
import logging

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class RerankService:
    """
    Cross-encoder reranking of retrieval candidates.

    Unlike a bi-encoder (used for the initial dense retrieval, where query
    and document are embedded independently), a cross-encoder scores the
    query and document together in one forward pass. That's far more
    accurate at judging relevance but too slow to run over an entire
    collection - so it's used as a second-pass re-ranker over a small
    candidate pool produced by the first-pass retrieval (dense or hybrid),
    not as the primary search mechanism.
    """

    def __init__(self, model_name: str):
        """
        Args:
            model_name: HuggingFace cross-encoder model name
                       (e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2")
        """
        self.model = CrossEncoder(model_name)
        logger.info(f"Loaded reranking model: {model_name}")

    def rerank(self, query: str, results, top_k: int):
        """
        Re-score and re-order retrieval results by relevance to the query,
        trimming to top_k.

        Args:
            query: The original search query
            results: Results in the standard nested-list shape
                     ({"ids": [[...]], "documents": [[...]], ...}) produced
                     by VectorService.search() / hybrid_search()
            top_k: Number of results to keep after reranking

        Returns:
            Results in the same nested-list shape, reordered and trimmed,
            with a "rerank_scores" key added (list of cross-encoder scores
            aligned to the returned documents)
        """
        ids = results["ids"][0] if results.get("ids") else []
        documents = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        if not documents:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "rerank_scores": [[]],
            }

        pairs = [[query, document] for document in documents]
        scores = self.model.predict(pairs)

        order = sorted(range(len(documents)), key=lambda i: scores[i], reverse=True)[:top_k]

        return {
            "ids": [[ids[i] for i in order]],
            "documents": [[documents[i] for i in order]],
            "metadatas": [[metadatas[i] for i in order]],
            "distances": [[distances[i] for i in order]],
            "rerank_scores": [[float(scores[i]) for i in order]],
        }
