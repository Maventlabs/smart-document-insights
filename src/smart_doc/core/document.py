"""Document loading and statistics."""

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from smart_doc.config import SUPPORTED_FILE_TYPES


def load_documents(file_paths: list[str]) -> list:
    """Load documents from a list of file paths based on their extension.

    Supports: PDF, TXT, DOCX

    Args:
        file_paths: List of file paths to load.

    Returns:
        List of LangChain Document objects.

    Raises:
        ValueError: If a file type is not supported.
    """
    all_docs = []
    for path in file_paths:
        import os
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext == ".txt":
            loader = TextLoader(path, encoding="utf-8")
        elif ext == ".docx":
            loader = Docx2txtLoader(path)
        else:
            raise ValueError(f"Tipe file tidak didukung: {ext}")
        all_docs.extend(loader.load())
    return all_docs


def _count_words(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())


def get_doc_stats(documents: list) -> dict:
    """Calculate statistics from loaded documents.

    Args:
        documents: List of LangChain Document objects.

    Returns:
        Dict with keys: pages, words, chars.
    """
    total_pages = len(documents)
    total_text = "\n".join([doc.page_content for doc in documents])
    total_words = _count_words(total_text)
    total_chars = len(total_text)
    return {
        "pages": total_pages,
        "words": total_words,
        "chars": total_chars,
    }
