"""Summary tab component for document summarization."""

from datetime import datetime
import streamlit as st
from smart_doc.core.rag import build_rag_chain, invoke_rag
from smart_doc.core.prompts import SYSTEM_PROMPT_SUMMARY


def render_summary_tab(vectorstore, llm, retriever_k: int):
    """Render the document summary tab.

    Args:
        vectorstore: Chroma vector store instance.
        llm: Language model instance.
        retriever_k: Number of chunks to retrieve.
    """
    st.subheader("Ringkasan Dokumen")
    st.markdown("Klik tombol di bawah untuk generate ringkasan otomatis dari seluruh dokumen.")

    if st.button("Generate Ringkasan", key="btn_summary", use_container_width=True):
        with st.spinner("Membuat ringkasan..."):
            try:
                rag_chain = build_rag_chain(vectorstore, llm, retriever_k, SYSTEM_PROMPT_SUMMARY)
                response = invoke_rag(rag_chain, "Buat ringkasan lengkap dari dokumen ini.")
                summary = response["answer"]

                st.markdown(summary)

                # Export summary
                summary_md = (
                    f"# Ringkasan Dokumen\n\n"
                    f"_Dibuat pada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n"
                    f"---\n\n{summary}"
                )
                st.download_button(
                    label="Export Ringkasan",
                    data=summary_md,
                    file_name=f"ringkasan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                )

            except Exception as e:
                st.error(f"Gagal membuat ringkasan: {e}")
