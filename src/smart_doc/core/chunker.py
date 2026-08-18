"""Semantic chunking with structure-aware splitting."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def semantic_chunk(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split documents into semantically meaningful chunks.

    Uses RecursiveCharacterTextSplitter with optimized separators
    that respect sentence and paragraph boundaries.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of chunked Documents with enhanced metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n\n",  # Section breaks
            "\n\n",    # Paragraph breaks
            "\n",      # Line breaks
            ". ",      # Sentence endings
            "! ",      # Exclamation
            "? ",      # Question
            "; ",      # Semicolon
            ", ",      # Comma
            " ",       # Word boundary
            "",        # Character level
        ],
        keep_separator=True,
    )

    chunks = text_splitter.split_documents(documents)

    # Enhance chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        chunk.metadata["chunk_word_count"] = len(chunk.page_content.split())

        # Detect if chunk starts with a heading-like pattern
        first_line = chunk.page_content.split("\n")[0].strip()
        if _is_heading(first_line):
            chunk.metadata["section_title"] = first_line

    return chunks


def _is_heading(text: str) -> bool:
    """Check if text looks like a heading."""
    import re
    patterns = [
        re.compile(r"^(PASAL|BAB|BAGIAN|SECTION|CHAPTER|ARTICLE)\s+\d+", re.IGNORECASE),
        re.compile(r"^\d+[\.\)]\s+[A-Z][A-Za-z\s]+"),
        re.compile(r"^[A-Z][A-Z\s]{5,}$"),
    ]
    for p in patterns:
        if p.search(text):
            return True
    return False
