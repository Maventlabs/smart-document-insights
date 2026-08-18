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
    """Render the sidebar and return configuration values."""
    with st.sidebar:
        # ── Logo / Title ──
        st.markdown(
            """
            <div style="padding: 0.25rem 0 0.75rem 0;">
                <div style="font-size:0.75rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; font-weight:500;">Config</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── API Key ──
        nim_api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="nvapi-...",
            help="build.nvidia.com -- gratis, no CC",
        )

        st.divider()

        # ── Model ──
        st.markdown(
            '<div style="font-size:0.75rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; font-weight:500; margin-bottom:0.5rem;">Model</div>',
            unsafe_allow_html=True,
        )
        model_names = [name for name, _ in NIM_CHAT_MODELS]
        model_ids = [mid for _, mid in NIM_CHAT_MODELS]
        model_index = st.selectbox(
            "",
            options=model_names,
            index=0,
            label_visibility="collapsed",
        )
        model_choice = model_ids[model_names.index(model_index)]

        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)

        st.divider()

        # ── Upload ──
        st.markdown(
            '<div style="font-size:0.75rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; font-weight:500; margin-bottom:0.5rem;">Dokumen</div>',
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "",
            type=SUPPORTED_FILE_TYPES,
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="PDF, TXT, DOCX",
        )

        st.divider()

        # ── Advanced ──
        with st.expander("Advanced"):
            chunk_size = st.slider("Chunk size", 200, 4000, CHUNK_SIZE_DEFAULT, 100)
            chunk_overlap = st.slider("Overlap", 0, 1000, CHUNK_OVERLAP_DEFAULT, 50)
            retriever_k = st.slider("Top-k", 1, 15, RETRIEVER_K_DEFAULT)

    return {
        "nim_api_key": nim_api_key,
        "model_choice": model_choice,
        "temperature": temperature,
        "uploaded_files": uploaded_files,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "retriever_k": retriever_k,
    }
