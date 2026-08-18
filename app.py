"""Smart Document Insights - Entry Point.

Advanced RAG pipeline:
  PDF -> Parse -> Structure -> Metadata -> Chunk -> Embed -> Index
  Query -> Rewrite -> [Semantic | BM25] -> Fusion -> Rerank -> Context -> LLM

Run with: streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from datetime import datetime

from smart_doc.core.pipeline import RAGPipeline
from smart_doc.core.rag import build_llm
from smart_doc.core.document import load_documents, get_doc_stats
from smart_doc.core.prompts import SYSTEM_PROMPT_DEFAULT, SYSTEM_PROMPT_SUMMARY, SYSTEM_PROMPT_INSIGHTS
from smart_doc.utils.file import save_uploaded_file, cleanup_file, validate_file_types
from smart_doc.utils.export import export_chat_to_markdown
from smart_doc.ui.sidebar import render_sidebar
from smart_doc.ui.stats import render_document_stats
from smart_doc.config import SUPPORTED_FILE_TYPES


def inject_dark_theme():
    """Inject dark theme CSS. Anti-AI-slop: minimal, clean, no gradients."""
    st.markdown(
        """
        <style>
        /* ── Base Dark Theme ────────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');

        :root {
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --border: #30363d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --accent: #58a6ff;
            --accent-dim: #1f6feb;
            --green: #3fb950;
            --red: #f85149;
            --yellow: #d29922;
        }

        /* App background */
        .stApp, [data-testid="stAppViewContainer"], .main .block-container {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }

        /* Sidebar dark panel */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div {
            background-color: var(--bg-secondary) !important;
            border-right: 1px solid var(--border) !important;
        }

        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] .stMarkdown li,
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] label {
            color: var(--text-secondary) !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: var(--text-primary) !important;
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
            font-family: 'Inter', sans-serif !important;
        }

        /* Body text */
        p, li, span, label, div {
            color: var(--text-primary) !important;
        }

        /* Inputs */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > div,
        .stNumberInput > div > div > input {
            background-color: var(--bg-tertiary) !important;
            border: 1px solid var(--border) !important;
            color: var(--text-primary) !important;
            border-radius: 6px !important;
        }

        /* Sliders */
        .stSlider > div > div > div > div {
            background-color: var(--accent) !important;
        }

        /* Buttons */
        .stButton > button {
            background-color: var(--accent-dim) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 6px !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
            transition: background 0.15s ease !important;
        }
        .stButton > button:hover {
            background-color: var(--accent) !important;
        }

        /* Download button */
        .stDownloadButton > button {
            background-color: transparent !important;
            color: var(--accent) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
        }
        .stDownloadButton > button:hover {
            background-color: var(--bg-tertiary) !important;
        }

        /* File uploader */
        [data-testid="stFileUploader"] {
            border: 2px dashed var(--border) !important;
            border-radius: 8px !important;
            padding: 1rem !important;
        }
        [data-testid="stFileUploader"] label {
            color: var(--text-secondary) !important;
        }

        /* Metrics */
        div[data-testid="stMetric"] {
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            padding: 0.75rem 1rem !important;
        }
        div[data-testid="stMetric"] label {
            color: var(--text-secondary) !important;
            font-size: 0.8rem !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: var(--text-primary) !important;
        }

        /* Chat messages */
        div[data-testid="stChatMessage"] {
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            padding: 1rem 1.25rem !important;
            margin-bottom: 0.75rem !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: var(--bg-secondary) !important;
            border-radius: 8px !important;
            padding: 4px !important;
            gap: 2px !important;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent !important;
            color: var(--text-secondary) !important;
            border-radius: 6px !important;
            padding: 8px 16px !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: var(--bg-tertiary) !important;
            color: var(--text-primary) !important;
        }

        /* Expander */
        .streamlit-expanderHeader {
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            color: var(--text-secondary) !important;
        }

        /* Dividers */
        hr {
            border-color: var(--border) !important;
        }

        /* Status messages */
        .stAlert > div {
            border-radius: 6px !important;
        }

        /* Chat input */
        .stChatInput > div {
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }

        /* Hide Streamlit branding */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }

        /* Code blocks */
        code {
            font-family: 'JetBrains Mono', monospace !important;
            background-color: var(--bg-tertiary) !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            font-size: 0.85em !important;
        }
        pre {
            background-color: var(--bg-tertiary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            padding: 1rem !important;
        }
        pre code {
            background-color: transparent !important;
            padding: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="Smart Document Insights",
        page_icon="\u25b3",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_dark_theme()

    # Minimal header
    st.markdown(
        """
        <div style="margin-bottom: 0.5rem;">
            <h1 style="margin:0; font-size:1.5rem; letter-spacing:-0.02em;">
                <span style="color:#58a6ff;">\u25b3</span> Smart Document Insights
            </h1>
            <p style="color:#8b949e; margin:0.25rem 0 0 0; font-size:0.875rem;">
                Hybrid RAG. Vector + BM25 + Reranker. Powered by NVIDIA NIM.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
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
        st.warning("Masukkan NVIDIA NIM API Key di sidebar. Gratis di build.nvidia.com")
        st.stop()

    # Validation: Files
    if not uploaded_files:
        st.info("Upload dokumen di sidebar untuk mulai.")
        st.stop()

    # Validate file types
    invalid_files = validate_file_types(uploaded_files, SUPPORTED_FILE_TYPES)
    if invalid_files:
        names = ", ".join([f.name for f in invalid_files])
        st.error(f"File tidak didukung: {names}. Pakai: {', '.join(SUPPORTED_FILE_TYPES)}")
        st.stop()

    # Process documents
    file_paths = []
    pipeline = None
    chunks = []
    doc_metadata_list = []
    try:
        for uf in uploaded_files:
            file_paths.append(save_uploaded_file(uf))

        with st.spinner("Parse -> Structure -> Chunk -> Embed -> Index..."):
            try:
                pipeline = RAGPipeline(
                    nim_api_key=nim_api_key,
                    model=model_choice,
                    temperature=temperature,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    retriever_k=retriever_k,
                )
                vectorstore, chunks, doc_metadata_list = pipeline.process_documents(file_paths)
                st.success(f"{len(chunks)} chunks dari {len(uploaded_files)} file")

                # Compact pipeline info
                with st.expander("Dokumen & Pipeline"):
                    for meta in doc_metadata_list:
                        fname = os.path.basename(meta.source)
                        st.markdown(
                            f"**{fname}** -- {meta.page_count} halaman, "
                            f"{meta.total_words:,} kata, bahasa: {meta.language}"
                        )
            except Exception as e:
                st.error(f"Gagal: {e}")
                st.stop()
    finally:
        for fp in file_paths:
            cleanup_file(fp)

    # Document Stats
    raw_docs = load_documents(file_paths if file_paths else [])
    doc_stats = get_doc_stats(raw_docs)
    render_document_stats(doc_stats, len(chunks))

    st.divider()

    # Tabs
    tab_chat, tab_summary, tab_insights = st.tabs(["Chat", "Ringkasan", "Insights"])

    llm = build_llm(model_choice, temperature, nim_api_key)

    with tab_chat:
        render_chat_tab(vectorstore, llm, retriever_k, uploaded_files, pipeline=pipeline)

        if st.session_state.get("messages"):
            md = export_chat_to_markdown(st.session_state.messages)
            st.download_button(
                label="Export",
                data=md,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )

    with tab_summary:
        render_summary_tab(vectorstore, llm, retriever_k)

    with tab_insights:
        render_insights_tab(vectorstore, llm, retriever_k)


if __name__ == "__main__":
    main()
