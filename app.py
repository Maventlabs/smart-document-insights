"""Smart Document Insights - Entry Point.

Run with: streamlit run app.py
"""

import sys
import os

# Ensure src/ is on the Python path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from datetime import datetime

from smart_doc.core.embeddings import process_documents
from smart_doc.core.rag import build_llm
from smart_doc.utils.file import save_uploaded_file, cleanup_file, validate_file_types
from smart_doc.utils.export import export_chat_to_markdown, export_to_markdown
from smart_doc.ui.sidebar import render_sidebar
from smart_doc.ui.stats import render_document_stats
from smart_doc.ui.chat import render_chat_tab
from smart_doc.ui.summary import render_summary_tab
from smart_doc.ui.insights import render_insights_tab
from smart_doc.core.document import get_doc_stats
from smart_doc.config import SUPPORTED_FILE_TYPES


def main():
    st.set_page_config(
        page_title="Smart Document Insights",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        .stApp { max-width: 100%; }
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        div[data-testid="stMetric"] {
            background-color: #f0f2f6;
            border-radius: 0.5rem;
            padding: 0.75rem 1rem;
        }
        div[data-testid="stMetric"] label { font-size: 0.85rem !important; }
        div[data-testid="stChatMessage"] {
            border-radius: 0.75rem;
            padding: 0.5rem 1rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header
    st.title("Smart Document Insights")
    st.markdown(
        "**Alat berbasis AI** yang memahami dokumen kompleks -- kontrak hukum, makalah penelitian, "
        "laporan keuangan, dan lainnya. Upload -> Tanya -> Dapatkan Jawaban.\n"
        "*Didukung oleh [NVIDIA NIM](https://build.nvidia.com) -- 100+ model AI gratis.*"
    )

    # Sidebar
    cfg = render_sidebar()

    nim_api_key = cfg["nim_api_key"]
    model_choice = cfg["model_choice"]
    temperature = cfg["temperature"]
    uploaded_files = cfg["uploaded_files"]
    chunk_size = cfg["chunk_size"]
    chunk_overlap = cfg["chunk_overlap"]
    retriever_k = cfg["retriever_k"]

    # Validation: API Key
    if not nim_api_key:
        st.warning("Silakan masukkan **NVIDIA NIM API Key** di bilah samping untuk melanjutkan. (Gratis!)")
        st.stop()

    # Validation: Files
    if not uploaded_files:
        st.info("Silakan **unggah dokumen** (PDF/TXT/DOCX) di bilah samping untuk dianalisis.")
        st.stop()

    # Validate file types
    invalid_files = validate_file_types(uploaded_files, SUPPORTED_FILE_TYPES)
    if invalid_files:
        names = ", ".join([f.name for f in invalid_files])
        st.error(f"File tidak didukung: {names}. Format yang didukung: {', '.join(SUPPORTED_FILE_TYPES)}")
        st.stop()

    # Process documents
    file_paths = []
    try:
        for uf in uploaded_files:
            file_paths.append(save_uploaded_file(uf))

        with st.spinner("Memproses dokumen dan membuat vector database..."):
            try:
                vectorstore, raw_documents, chunks = process_documents(
                    tuple(file_paths),
                    chunk_size,
                    chunk_overlap,
                    nim_api_key,
                )
                st.success(f"Berhasil memproses **{len(uploaded_files)} file** -> **{len(chunks)} chunks**")
            except Exception as e:
                st.error(f"Gagal memproses dokumen: {e}")
                st.stop()
    finally:
        for fp in file_paths:
            cleanup_file(fp)

    # Document Stats
    doc_stats = get_doc_stats(raw_documents)
    st.subheader("Statistik Dokumen")
    render_document_stats(doc_stats, len(chunks))

    # File list
    with st.expander("File yang diunggah"):
        for i, uf in enumerate(uploaded_files, 1):
            file_size = uf.size / 1024
            st.markdown(f"{i}. **{uf.name}** ({file_size:.1f} KB)")

    st.divider()

    # Tabs
    tab_chat, tab_summary, tab_insights = st.tabs([
        "Tanya Jawab",
        "Ringkasan Dokumen",
        "Key Insights",
    ])

    # Build LLM
    llm = build_llm(model_choice, temperature, nim_api_key)

    with tab_chat:
        render_chat_tab(vectorstore, llm, retriever_k, uploaded_files)

        # Export chat
        if st.session_state.get("messages"):
            md = export_chat_to_markdown(st.session_state.messages)
            st.download_button(
                label="Export Chat History",
                data=md,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )

    with tab_summary:
        render_summary_tab(vectorstore, llm, retriever_k)

    with tab_insights:
        render_insights_tab(vectorstore, llm, retriever_k)


if __name__ == "__main__":
    main()
