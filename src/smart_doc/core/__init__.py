"""Core business logic for document processing and RAG."""

from smart_doc.core.document import load_documents, get_doc_stats
from smart_doc.core.embeddings import process_documents, create_embeddings, create_vectorstore
from smart_doc.core.rag import build_llm, build_rag_chain, invoke_rag
from smart_doc.core.prompts import (
    SYSTEM_PROMPT_DEFAULT,
    SYSTEM_PROMPT_SUMMARY,
    SYSTEM_PROMPT_INSIGHTS,
)
from smart_doc.core.pipeline import RAGPipeline
from smart_doc.core.document_parser import parse_document, DocumentMetadata
from smart_doc.core.chunker import semantic_chunk
from smart_doc.core.retriever import HybridRetriever
from smart_doc.core.reranker import CrossEncoderReranker, SimpleReranker
from smart_doc.core.query_rewriter import rewrite_query, expand_query

__all__ = [
    "load_documents",
    "get_doc_stats",
    "process_documents",
    "create_embeddings",
    "create_vectorstore",
    "build_llm",
    "build_rag_chain",
    "invoke_rag",
    "SYSTEM_PROMPT_DEFAULT",
    "SYSTEM_PROMPT_SUMMARY",
    "SYSTEM_PROMPT_INSIGHTS",
    "RAGPipeline",
    "parse_document",
    "DocumentMetadata",
    "semantic_chunk",
    "HybridRetriever",
    "CrossEncoderReranker",
    "SimpleReranker",
    "rewrite_query",
    "expand_query",
]
