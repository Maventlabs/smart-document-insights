"""Insights tab component for key insights extraction."""

from datetime import datetime
import streamlit as st
from smart_doc.core.rag import build_rag_chain, invoke_rag
from smart_doc.core.prompts import SYSTEM_PROMPT_INSIGHTS


def render_insights_tab(vectorstore, llm, retriever_k: int):
    """Render the key insights tab.

    Args:
        vectorstore: Chroma vector store instance.
        llm: Language model instance.
        retriever_k: Number of chunks to retrieve.
    """
    st.subheader("Key Insights")
    st.markdown("Ekstrak wawasan kunci dari dokumen -- temuan utama, data penting, risiko, dan rekomendasi.")

    insight_topic = st.text_input(
        "Topik spesifik (opsional)",
        placeholder="Contoh: analisis risiko, perbandingan Q1 vs Q2, klausul perjanjian...",
        help="Kosongkan untuk analisis umum, atau masukkan topik spesifik.",
    )

    if st.button("Ekstrak Insights", key="btn_insights", use_container_width=True):
        query = (
            f"Ekstrak key insights dari dokumen ini terkait: {insight_topic}"
            if insight_topic
            else "Ekstrak key insights utama dari seluruh dokumen ini."
        )
        with st.spinner("Menganalisis dokumen..."):
            try:
                rag_chain = build_rag_chain(vectorstore, llm, retriever_k, SYSTEM_PROMPT_INSIGHTS)
                response = invoke_rag(rag_chain, query)
                insights = response["answer"]

                st.markdown(insights)

                # Export insights
                insights_md = (
                    f"# Key Insights\n\n"
                    f"_Dibuat pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n"
                    f"---\n\n{insights}"
                )
                st.download_button(
                    label="Export Insights",
                    data=insights_md,
                    file_name=f"insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                )

            except Exception as e:
                st.error(f"Gagal mengekstrak insights: {e}")
