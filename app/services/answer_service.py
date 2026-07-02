
import logging

from google import genai

from app.services.vector_service import VectorService

logger = logging.getLogger(__name__)


class AnswerService:
    """
    Retrieval-augmented generation: retrieves context chunks via
    VectorService.retrieve() (dense, hybrid, and/or reranked - the same
    retrieval stack /vector/search uses, so this endpoint isn't a separate
    implementation) and asks a Gemini model to answer strictly from that
    context, with inline citations back to the source chunks.

    Uses Google AI Studio's Gemini API specifically because it has a free
    tier - unlike most hosted LLM APIs, this endpoint is usable without
    billing. Get a free key at https://aistudio.google.com/apikey
    """

    ANSWER_INSTRUCTION = (
        "Answer the question using ONLY the numbered context chunks below. "
        "Cite every claim with the bracketed chunk number(s) that support "
        "it, e.g. \"Diabetes symptoms include fatigue [1][3].\" If the "
        "context does not contain enough information to answer, say so "
        "explicitly instead of guessing."
    )

    def __init__(self, vector_service: VectorService, api_key: str, model: str):
        self.vector_service = vector_service
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        if not self.api_key:
            raise RuntimeError(
                "RAG endpoint not configured: set GOOGLE_API_KEY "
                "(get a free key at https://aistudio.google.com/apikey)"
            )
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def ask(self, query: str, filters, top_k: int, hybrid: bool = False, rerank: bool = False):
        """
        Retrieve context and generate a cited answer.

        Args:
            query: The user's question
            filters: Optional SearchFilters for metadata-scoped retrieval
            top_k: Number of context chunks to retrieve
            hybrid: Use dense+BM25 hybrid retrieval instead of dense-only
            rerank: Apply cross-encoder reranking to the retrieved chunks

        Returns:
            {"answer": str, "citations": [{index, document_id,
             document_name, page_number, chunk_text, distance}, ...]}
        """
        # Fail fast on missing config before doing any retrieval work
        client = self._get_client()

        results = self.vector_service.retrieve(query, filters, top_k, hybrid=hybrid, rerank=rerank)

        documents = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        if not documents:
            return {
                "answer": "No indexed documents matched this query, so there's no context to answer from.",
                "citations": [],
            }

        context_block = "\n\n".join(
            f"[{i + 1}] {document}" for i, document in enumerate(documents)
        )
        prompt = f"{self.ANSWER_INSTRUCTION}\n\nContext:\n{context_block}\n\nQuestion: {query}"

        response = client.models.generate_content(model=self.model, contents=prompt)

        citations = [
            {
                "index": i + 1,
                "document_id": metadatas[i].get("document_id"),
                "document_name": metadatas[i].get("document_name"),
                "page_number": metadatas[i].get("page_number"),
                "chunk_text": documents[i],
                "distance": distances[i] if i < len(distances) else None,
            }
            for i in range(len(documents))
        ]

        return {"answer": response.text, "citations": citations}
