import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

@st.cache_resource
def process_document(uploaded_file, openai_api_key):
    # Simpan file sementara untuk PyPDFLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # 1. Load document
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()

        # 2. Split document menjadi chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        # 3. Buat embeddings dan simpan ke Chroma vector store
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            persist_directory=None # In-memory untuk saat ini
        )
        return vectorstore
    finally:
        # Bersihkan file sementara
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def main():
    st.set_page_config(page_title="Smart Document Insights", page_icon="📄", layout="wide")
    st.title("Smart Document Insights 📄")
    st.markdown("Alat berbasis AI yang mampu memahami dokumen berukuran besar atau kompleks, seperti kontrak hukum, makalah penelitian, dan laporan keuangan.")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Konfigurasi")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        st.markdown("[Dapatkan OpenAI API key di sini](https://platform.openai.com/account/api-keys)")
        
        st.header("Unggah Dokumen")
        uploaded_file = st.file_uploader("Upload PDF Anda", type="pdf")
        
    if not openai_api_key:
        st.warning("Silakan masukkan OpenAI API Key di bilah samping untuk melanjutkan.")
        st.stop()
        
    if not uploaded_file:
        st.info("Silakan unggah dokumen PDF di bilah samping untuk dianalisis.")
        st.stop()
        
    with st.spinner("Memproses dokumen dan membuat vector database..."):
        try:
            vectorstore = process_document(uploaded_file, openai_api_key)
            st.success("Dokumen berhasil diproses dan dimuat ke dalam Vector Database!")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses dokumen: {e}")
            st.stop()

    # Chat interface
    st.header("Ajukan Pertanyaan")
    
    # Inisialisasi chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Tampilkan chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input pertanyaan dari user
    if prompt := st.chat_input("Apa yang ingin Anda ketahui dari dokumen ini?"):
        # Tambahkan pertanyaan user ke history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                try:
                    # Setup LLM
                    llm = ChatOpenAI(
                        model="gpt-3.5-turbo",
                        temperature=0,
                        openai_api_key=openai_api_key
                    )

                    # Setup Prompt
                    system_prompt = (
                        "Anda adalah asisten AI yang ahli dalam menganalisis dokumen kompleks seperti kontrak, "
                        "makalah, dan laporan keuangan.\n"
                        "Gunakan potongan konteks berikut untuk menjawab pertanyaan.\n"
                        "Jika Anda tidak tahu jawabannya, katakan saja bahwa Anda tidak tahu berdasarkan dokumen ini.\n"
                        "Berikan jawaban yang komprehensif, terstruktur, dan mudah dipahami.\n\n"
                        "Konteks: {context}"
                    )
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])

                    # Setup Retrieval Chain
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
                    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                    # Generate jawaban
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    
                    st.markdown(answer)
                    
                    # Tambahkan jawaban assistant ke history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Tampilkan sumber dokumen yang relevan (opsional, bisa di-expand)
                    with st.expander("Lihat Sumber Konteks"):
                        for i, doc in enumerate(response["context"]):
                            st.markdown(f"**Potongan {i+1}** (Halaman {doc.metadata.get('page', 'Unknown')}):")
                            st.text(doc.page_content)
                            st.markdown("---")

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat menghasilkan jawaban: {e}")

if __name__ == '__main__':
    main()
