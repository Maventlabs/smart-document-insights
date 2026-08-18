"""Document statistics display component."""

import streamlit as st


def render_document_stats(doc_stats: dict, chunks_count: int):
    """Render document statistics in a metric row.

    Args:
        doc_stats: Dict with 'pages', 'words', 'chars' keys.
        chunks_count: Number of text chunks created.
    """
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Halaman", doc_stats["pages"])
    col2.metric("Kata", f"{doc_stats['words']:,}")
    col3.metric("Karakter", f"{doc_stats['chars']:,}")
    col4.metric("Chunks", chunks_count)
