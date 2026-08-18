"""Hybrid retriever combining vector search (semantic) and BM25 (keyword)."""

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class HybridRetriever:
    """Combines semantic (vector) retrieval with BM25 keyword retrieval.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both retrievers.
    """

    def __init__(self, vectorstore, chunks: list[Document], k: int = 5):
        """Initialize the hybrid retriever.

        Args:
            vectorstore: Chroma vector store for semantic search.
            chunks: Original document chunks for BM25 indexing.
            k: Number of results to return.
        """
        self.vectorstore = vectorstore
        self.k = k

        # Build BM25 index
        self.chunks = chunks
        tokenized_corpus = [self._tokenize(doc.page_content) for doc in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, chat_history: list[dict] = None) -> list[Document]:
        """Retrieve relevant documents using hybrid search.

        Args:
            query: The search query.
            chat_history: Previous chat messages for context.

        Returns:
            List of relevant Document objects ranked by relevance.
        """
        # Semantic retrieval (vector search)
        semantic_results = self.vectorstore.similarity_search_with_score(query, k=self.k * 2)

        # Keyword retrieval (BM25)
        bm25_scores = self.bm25.get_scores(self._tokenize(query))
        bm25_top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:self.k * 2]

        bm25_results = [
            (self.chunks[i], bm25_scores[i])
            for i in bm25_top_indices
            if bm25_scores[i] > 0
        ]

        # Fuse results using Reciprocal Rank Fusion (RRF)
        fused = self._reciprocal_rank_fusion(semantic_results, bm25_results, k=60)

        # Return top-k
        return fused[:self.k]

    def _reciprocal_rank_fusion(
        self,
        semantic_results: list,
        bm25_results: list,
        k: int = 60,
    ) -> list[Document]:
        """Merge results using Reciprocal Rank Fusion.

        RRF formula: score = sum(1 / (k + rank)) across all result lists.

        Args:
            semantic_results: Results from vector search (doc, score) tuples.
            bm25_results: Results from BM25 search (doc, score) tuples.
            k: RRF constant (default 60).

        Returns:
            Fused and ranked list of Documents.
        """
        doc_scores = {}

        # Process semantic results
        for rank, (doc, score) in enumerate(semantic_results):
            doc_id = self._doc_id(doc)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0}
            doc_scores[doc_id]["score"] += 1.0 / (k + rank + 1)

        # Process BM25 results
        for rank, (doc, score) in enumerate(bm25_results):
            doc_id = self._doc_id(doc)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"doc": doc, "score": 0}
            doc_scores[doc_id]["score"] += 1.0 / (k + rank + 1)

        # Sort by fused score
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs]

    def _doc_id(self, doc: Document) -> str:
        """Generate a unique ID for a document chunk."""
        page = doc.metadata.get("page", "")
        source = doc.metadata.get("source", "")
        content_hash = hash(doc.page_content[:100])
        return f"{source}:{page}:{content_hash}"

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25 (simple whitespace + lowercase)."""
        import re
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()
