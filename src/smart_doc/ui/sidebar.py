"""Sidebar component for configuration and file upload."""

import streamlit as st
from smart_doc.config import (
    SUPPORTED_FILE_TYPES,
    CHUNK_SIZE_DEFAULT,
    CHUNK_OVERLAP_DEFAULT,
    RETRIEVER_K_DEFAULT,
    NIM_CHAT_MODELS,
)


def render_sidebar() -> dict:
    """Render the sidebar and return configuration values.

    Returns:
        Dict with keys: nim_api_key, model_choice, temperature,
        uploaded_files, chunk_size, chunk_overlap, retriever_k.
    """
    with st.sidebar:
        st.header("Konfigurasi")

        nim_api_key = st.text_input(
            "NVIDIA NIM API Key",
            type="password",
            help="Masukkan API key Anda dari build.nvidia.com (Gratis!)",
        )
        st.markdown("[Dapatkan API key gratis](https://build.nvidia.com)")
        st.caption("Gratis 1.000+ inference credits. Tanpa kartu kredit.")

        st.divider()

        # Model selection
        st.header("Model AI")
        model_names = [name for name, _ in NIM_CHAT_MODELS]
        model_ids = [mid for _, mid in NIM_CHAT_MODELS]
        model_index = st.selectbox(
            "Pilih Model",
            options=model_names,
            index=0,
            help="Model lebih besar = lebih akurat, lebih lambat. Semua gratis via NVIDIA NIM.",
        )
        model_choice = model_ids[model_names.index(model_index)]

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Semakin rendah, semakin fokus. Semakin tinggi, semakin kreatif.",
        )

        st.divider()

        # Document upload
        st.header("Unggah Dokumen")
        uploaded_files = st.file_uploader(
            "Upload file Anda",
            type=SUPPORTED_FILE_TYPES,
            accept_multiple_files=True,
            help="Format: PDF, TXT, DOCX. Bisa unggah beberapa file sekaligus.",
        )

        st.divider()

        # Advanced settings
        st.header("Pengaturan Lanjut")
        chunk_size = st.slider(
            "Ukuran Chunk",
            min_value=200,
            max_value=4000,
            value=CHUNK_SIZE_DEFAULT,
            step=100,
            help="Ukuran potongan teks. Chunk lebih kecil = lebih detail, lebih banyak chunks.",
        )
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=1000,
            value=CHUNK_OVERLAP_DEFAULT,
            step=50,
            help="Tumpang tindih antar chunks untuk menjaga konteks.",
        )
        retriever_k = st.slider(
            "Jumlah Konteks (k)",
            min_value=1,
            max_value=15,
            value=RETRIEVER_K_DEFAULT,
            help="Jumlah chunk yang diambil sebagai konteks untuk menjawab.",
        )

    return {
        "nim_api_key": nim_api_key,
        "model_choice": model_choice,
        "temperature": temperature,
        "uploaded_files": uploaded_files,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "retriever_k": retriever_k,
    }
