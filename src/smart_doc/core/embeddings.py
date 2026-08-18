"""Embeddings and vector store processing using NVIDIA NIM."""

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from smart_doc.core.document import load_documents
from smart_doc.config import (
    NIM_BASE_URL,
    NIM_EMBEDDING_MODEL,
    CHUNK_SIZE_DEFAULT,
    CHUNK_OVERLAP_DEFAULT,
)


@st.cache_resource(show_spinner=False)
def process_documents(
    _file_paths: tuple,
    chunk_size: int = CHUNK_SIZE_DEFAULT,
    chunk_overlap: int = CHUNK_OVERLAP_DEFAULT,
    nim_api_key: str = "",
):
    """Process multiple documents: load -> split -> embed -> vector store.

    Note: _file_paths is prefixed with underscore so Streamlit won't try to hash it.

    Args:
        _file_paths: Tuple of file paths (for Streamlit caching).
        chunk_size: Size of text chunks.
        chunk_overlap: Overlap between chunks.
        nim_api_key: NVIDIA NIM API key.

    Returns:
        Tuple of (vectorstore, raw_documents, chunks).

    Raises:
        ValueError: If no documents or chunks can be created.
    """
    # Load documents
    documents = load_documents(list(_file_paths))
    if not documents:
        raise ValueError("Tidak ada konten yang bisa diekstrak dari file yang diunggah.")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Tidak ada chunk yang dihasilkan. Pastikan dokumen memiliki konten teks.")

    # Create embeddings using NVIDIA NIM
    embeddings = OpenAIEmbeddings(
        model=NIM_EMBEDDING_MODEL,
        base_url=NIM_BASE_URL,
        api_key=nim_api_key,
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=None,  # In-memory
    )
    return vectorstore, documents, chunks
