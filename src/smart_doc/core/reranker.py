"""Reranker for precision retrieval. Lightweight BM25-based."""

import re
from langchain_core.documents import Document


# Stop words for reranking
STOP_WORDS = {
    "yang", "dan", "ini", "itu", "dengan", "untuk", "pada", "adalah",
    "akan", "tidak", "dari", "dalam", "oleh", "atau", "juga", "saya",
    "the", "a", "an", "is", "are", "was", "in", "on", "of", "to",
    "and", "or", "not", "it", "for", "with", "by",
}


class BM25Reranker:
    """Lightweight reranker using BM25 scoring + TF-IDF."""

    def __init__(self):
        pass

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """Rerank documents using BM25-like scoring.

        Combines:
        - TF-IDF term overlap
        - Phrase matching bonus
        - Document length normalization
        """
        if not documents:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return documents[:top_k]

        scored = []
        for doc in documents:
            doc_terms = self._tokenize(doc.page_content)
            score = self._bm25_score(query_terms, doc_terms, len(documents))
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_k]]

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text, removing stop words and short tokens."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]

    def _bm25_score(self, query_terms: list[str], doc_terms: list[str], total_docs: int) -> float:
        """Calculate BM25-like score."""
        if not doc_terms:
            return 0.0

        doc_len = len(doc_terms)
        doc_term_set = {}
        for t in doc_terms:
            doc_term_set[t] = doc_term_set.get(t, 0) + 1

        score = 0.0
        k1 = 1.5  # Term frequency saturation
        b = 0.75  # Length normalization
        avg_dl = max(doc_len, 1)

        for qt in query_terms:
            if qt in doc_term_set:
                tf = doc_term_set[qt]
                # BM25 term score
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / avg_dl)
                score += numerator / denominator

            # Phrase bonus: check if query term appears as substring
            doc_text = " ".join(doc_terms)
            if qt in doc_text:
                score += 0.1

        # Normalize by query length
        if query_terms:
            score /= len(query_terms)

        return score
