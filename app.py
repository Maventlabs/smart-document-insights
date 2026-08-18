"""Maventrag - Entry Point.

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
from smart_doc.core.document import get_doc_stats
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
        }

        .stApp, [data-testid="stAppViewContainer"], .main .block-container {
            background-color: var(--bg-primary) !important;
            color: var(--text-primary) !important;
        }

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

        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
            font-family: 'Inter', sans-serif !important;
        }

        p, li, span, label, div {
            color: var(--text-primary) !important;
        }

        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > div,
        .stNumberInput > div > div > input {
            background-color: var(--bg-tertiary) !important;
            border: 1px solid var(--border) !important;
            color: var(--text-primary) !important;
            border-radius: 6px !important;
        }

        .stSlider > div > div > div > div {
            background-color: var(--accent) !important;
        }

        .stButton > button {
            background-color: var(--accent-dim) !important;
            color: #fff !important;
            border: none !important;
            border-radius: 6px !important;
            font-family: 'Inter', sans-serif !important;
            font-weight: 500 !important;
        }
        .stButton > button:hover {
            background-color: var(--accent) !important;
        }

        .stDownloadButton > button {
            background-color: transparent !important;
            color: var(--accent) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
        }
        .stDownloadButton > button:hover {
            background-color: var(--bg-tertiary) !important;
        }

        [data-testid="stFileUploader"] {
            border: 2px dashed var(--border) !important;
            border-radius: 8px !important;
            padding: 1rem !important;
        }

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

        div[data-testid="stChatMessage"] {
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            padding: 1rem 1.25rem !important;
            margin-bottom: 0.75rem !important;
        }

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

        .streamlit-expanderHeader {
            background-color: var(--bg-secondary) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            color: var(--text-secondary) !important;
        }

        hr { border-color: var(--border) !important; }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }

        code {
            font-family: 'JetBrains Mono', monospace !important;
            background-color: var(--bg-tertiary) !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
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
        page_title="Maventrag",
        page_icon="\u25b3",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_dark_theme()

    st.markdown(
        """
        <div style="margin-bottom: 0.5rem;">
            <h1 style="margin:0; font-size:1.5rem; letter-spacing:-0.02em;">
                <span style="color:#58a6ff;">\u25b3</span> Maventrag
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

    # Validation
    if not nim_api_key:
        st.warning("Masukkan NVIDIA NIM API Key di sidebar. Gratis di build.nvidia.com")
        st.stop()

    if not uploaded_files:
        st.info("Upload dokumen di sidebar untuk mulai.")
        st.stop()

    invalid_files = validate_file_types(uploaded_files, SUPPORTED_FILE_TYPES)
    if invalid_files:
        names = ", ".join([f.name for f in invalid_files])
        st.error(f"File tidak didukung: {names}. Pakai: {', '.join(SUPPORTED_FILE_TYPES)}")
        st.stop()

    # Process documents
    file_paths = []
    vectorstore = None
    pipeline = None
    chunks = []
    doc_metadata_list = []
    pipeline_error = None

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
        except Exception as e:
            pipeline_error = str(e)

    # Clean up temp files
    for fp in file_paths:
        cleanup_file(fp)

    # If pipeline failed, show error and stop
    if pipeline_error:
        st.error(f"Gagal: {pipeline_error}")
        st.stop()

    if vectorstore is None:
        st.error("Pipeline gagal memproses dokumen.")
        st.stop()

    st.success(f"{len(chunks)} chunks dari {len(uploaded_files)} file")

    with st.expander("Dokumen & Pipeline"):
        for meta in doc_metadata_list:
            fname = os.path.basename(meta.source)
            st.markdown(
                f"**{fname}** -- {meta.page_count} halaman, "
                f"{meta.total_words:,} kata, bahasa: {meta.language}"
            )

    # Document Stats
    if doc_metadata_list:
        total_pages = sum(m.page_count for m in doc_metadata_list)
        total_words = sum(m.total_words for m in doc_metadata_list)
        total_chars = sum(m.total_chars for m in doc_metadata_list)
        doc_stats = {"pages": total_pages, "words": total_words, "chars": total_chars}
    else:
        doc_stats = {"pages": 0, "words": 0, "chars": 0}
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
