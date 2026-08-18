"""Document statistics display component."""

import streamlit as st


def render_document_stats(doc_stats: dict, chunks_count: int):
    """Render document statistics in a compact metric row."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Halaman", doc_stats["pages"])
    c2.metric("Kata", f"{doc_stats['words']:,}")
    c3.metric("Chars", f"{doc_stats['chars']:,}")
    c4.metric("Chunks", chunks_count)
