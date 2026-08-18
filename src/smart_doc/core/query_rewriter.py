"""Query rewriting and expansion for better retrieval."""


def rewrite_query(query: str, chat_history: list[dict] = None) -> str:
    """Rewrite a query for better retrieval.

    Handles:
    - Pronoun resolution (using chat history context)
    - Query expansion (adding related terms)
    - Query normalization

    Args:
        query: The original user query.
        chat_history: Previous chat messages for context.

    Returns:
        Rewritten query string.
    """
    rewritten = query.strip()

    # Add context from chat history for pronoun resolution
    if chat_history:
        last_exchange = chat_history[-2:] if len(chat_history) >= 2 else chat_history
        context = " ".join([m.get("content", "") for m in last_exchange])
        rewritten = _resolve_pronouns(rewritten, context)

    return rewritten


def expand_query(query: str) -> list[str]:
    """Expand a query into multiple variations for hybrid retrieval.

    Args:
        query: The original query.

    Returns:
        List of query variations (original + expansions).
    """
    queries = [query]

    # Add keyword-focused variation
    keywords = _extract_keywords(query)
    if keywords:
        queries.append(" ".join(keywords))

    # Add question reformulation
    if not query.strip().endswith("?"):
        queries.append(f"Apa itu {query}?")

    return queries


def _resolve_pronouns(query: str, context: str) -> str:
    """Simple pronoun resolution based on context."""
    pronouns = {
        "dia": _extract_subject(context),
        "itu": "",
        "ini": "",
    }
    # Simple heuristic: if query is very short and has pronouns, add context
    if len(query.split()) <= 3:
        for pronoun, replacement in pronouns.items():
            if pronoun in query.lower() and replacement:
                query = f"{replacement} {query}"
                break
    return query


def _extract_keywords(text: str) -> list[str]:
    """Extract important keywords from text."""
    # Remove common stop words
    stop_words = {
        "yang", "dan", "ini", "itu", "dengan", "untuk", "pada", "adalah",
        "akan", "tidak", "dari", "dalam", "oleh", "atau", "juga", "saya",
        "kami", "mereka", "bagaimana", "apa", "mengapa", "kapan", "dimana",
        "siapa", "berapa", "the", "a", "an", "is", "are", "was", "were",
        "this", "that", "with", "for", "in", "on", "by", "of", "to",
        "and", "or", "not", "it", "i", "we", "they", "how", "what",
        "why", "when", "where", "who", "which",
    }

    words = text.lower().split()
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    return keywords
