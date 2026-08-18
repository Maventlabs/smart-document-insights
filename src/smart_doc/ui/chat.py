"""Chat tab component for asking questions about documents."""

import streamlit as st
from smart_doc.core.rag import build_rag_chain, invoke_rag
from smart_doc.core.prompts import SYSTEM_PROMPT_DEFAULT


def render_chat_tab(vectorstore, llm, retriever_k: int, uploaded_files):
    """Render the chat Q&A tab.

    Args:
        vectorstore: Chroma vector store instance.
        llm: Language model instance.
        retriever_k: Number of chunks to retrieve.
        uploaded_files: List of uploaded file objects.
    """
    # Reset chat when files change
    file_names = tuple(sorted([f.name for f in uploaded_files]))
    if "last_files" not in st.session_state or st.session_state.last_files != file_names:
        st.session_state.messages = []
        st.session_state.last_files = file_names

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Apa yang ingin Anda ketahui dari dokumen ini?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                try:
                    rag_chain = build_rag_chain(vectorstore, llm, retriever_k, SYSTEM_PROMPT_DEFAULT)
                    response = invoke_rag(rag_chain, prompt)
                    answer = response["answer"]

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # Show sources
                    with st.expander("Sumber Konteks"):
                        for i, doc in enumerate(response["context"], 1):
                            page = doc.metadata.get("page", "N/A")
                            source = doc.metadata.get("source", "Unknown")
                            st.markdown(f"**Potongan {i}** -- Halaman: {page} | Sumber: `{source}`")
                            st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
                            st.markdown("---")

                except Exception as e:
                    error_msg = f"Terjadi kesalahan: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": f"Warning: {error_msg}"})
