"""Core business logic for document processing and RAG."""

from smart_doc.core.document import load_documents, get_doc_stats
from smart_doc.core.embeddings import process_documents
from smart_doc.core.rag import build_rag_chain, invoke_rag
from smart_doc.core.prompts import (
    SYSTEM_PROMPT_DEFAULT,
    SYSTEM_PROMPT_SUMMARY,
    SYSTEM_PROMPT_INSIGHTS,
)

__all__ = [
    "load_documents",
    "get_doc_stats",
    "process_documents",
    "build_rag_chain",
    "invoke_rag",
    "SYSTEM_PROMPT_DEFAULT",
    "SYSTEM_PROMPT_SUMMARY",
    "SYSTEM_PROMPT_INSIGHTS",
]
