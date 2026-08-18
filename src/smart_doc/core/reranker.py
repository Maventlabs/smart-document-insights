"""Cross-encoder reranker for precision retrieval."""

from langchain_core.documents import Document


class CrossEncoderReranker:
    """Rerank documents using a cross-encoder model.

    Uses sentence-transformers cross-encoder to score
    query-document relevance pairs for precise reranking.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the reranker.

        Args:
            model_name: HuggingFace cross-encoder model name.
                       Default: ms-marco-MiniLM-L-6-v2 (fast + accurate).
        """
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """Rerank documents based on query relevance.

        Args:
            query: The search query.
            documents: List of candidate Documents to rerank.
            top_k: Number of top results to return.

        Returns:
            Reranked list of Documents.
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [(query, doc.page_content) for doc in documents]

        # Score with cross-encoder
        scores = self.model.predict(pairs)

        # Sort by score
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return [doc for doc, score in scored_docs[:top_k]]


class SimpleReranker:
    """Lightweight reranker using BM25 scores (no GPU needed).

    Fallback option when sentence-transformers is not available.
    """

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """Simple reranking based on keyword overlap.

        Args:
            query: The search query.
            documents: List of candidate Documents to rerank.
            top_k: Number of top results to return.

        Returns:
            Reranked list of Documents.
        """
        import re

        query_words = set(re.findall(r"\w+", query.lower()))

        def score(doc: Document) -> float:
            doc_words = set(re.findall(r"\w+", doc.page_content.lower()))
            if not query_words:
                return 0
            overlap = query_words & doc_words
            return len(overlap) / len(query_words)

        scored = [(doc, score(doc)) for doc in documents]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored[:top_k]]
