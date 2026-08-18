"""Main RAG pipeline orchestrating all stages.

Pipeline flow:
  Document → Parse → Structure → Metadata → Chunk → Embed → Index

  User Query → Rewrite → [Semantic | BM25] → Fusion → Rerank → Context → LLM
"""

import streamlit as st
from langchain_core.documents import Document

from smart_doc.core.document_parser import parse_document, DocumentMetadata
from smart_doc.core.chunker import semantic_chunk
from smart_doc.core.embeddings import create_embeddings, create_vectorstore
from smart_doc.core.retriever import HybridRetriever
from smart_doc.core.reranker import CrossEncoderReranker, SimpleReranker
from smart_doc.core.query_rewriter import rewrite_query, expand_query
from smart_doc.core.rag import build_llm, build_rag_chain, invoke_rag
from smart_doc.core.prompts import SYSTEM_PROMPT_DEFAULT


class RAGPipeline:
    """Complete RAG pipeline with hybrid retrieval and reranking.

    Stages:
        1. Document parsing + metadata extraction
        2. Semantic chunking
        3. Embedding + vector indexing
        4. Query rewriting
        5. Hybrid retrieval (vector + BM25)
        6. Reranking
        7. Context selection
        8. LLM generation
    """

    def __init__(
        self,
        nim_api_key: str,
        model: str,
        temperature: float = 0.0,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        retriever_k: int = 5,
        use_cross_encoder: bool = True,
    ):
        """Initialize the RAG pipeline.

        Args:
            nim_api_key: NVIDIA NIM API key.
            model: Chat model identifier.
            temperature: LLM temperature.
            chunk_size: Text chunk size.
            chunk_overlap: Chunk overlap.
            retriever_k: Number of documents to retrieve.
            use_cross_encoder: Use cross-encoder reranker (slower but better).
        """
        self.nim_api_key = nim_api_key
        self.model = model
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.retriever_k = retriever_k

        # Components (initialized lazily)
        self._vectorstore = None
        self._chunks = None
        self._retriever = None
        self._reranker = None
        self._llm = None
        self._doc_metadata = None

        # Initialize reranker
        if use_cross_encoder:
            try:
                self._reranker = CrossEncoderReranker()
            except Exception:
                self._reranker = SimpleReranker()
        else:
            self._reranker = SimpleReranker()

    @st.cache_resource(show_spinner=False)
    def _build_vectorstore(_self, _file_paths: tuple):
        """Build vector store from file paths (cached)."""
        # Stage 1: Parse documents
        all_docs = []
        all_metadata = []
        for path in _file_paths:
            docs, meta = parse_document(path)
            all_docs.extend(docs)
            all_metadata.append(meta)

        # Stage 2: Semantic chunking
        chunks = semantic_chunk(
            all_docs,
            chunk_size=_self.chunk_size,
            chunk_overlap=_self.chunk_overlap,
        )

        # Stage 3: Embedding + indexing
        embeddings = create_embeddings(_self.nim_api_key)
        vectorstore = create_vectorstore(chunks, embeddings)

        return vectorstore, chunks, all_metadata

    def process_documents(self, file_paths: list[str]):
        """Process documents through the full pipeline.

        Args:
            file_paths: List of file paths to process.

        Returns:
            Tuple of (vectorstore, chunks, metadata_list).
        """
        self._vectorstore, self._chunks, self._doc_metadata = self._build_vectorstore(
            tuple(file_paths)
        )

        # Initialize hybrid retriever
        self._retriever = HybridRetriever(
            vectorstore=self._vectorstore,
            chunks=self._chunks,
            k=self.retriever_k,
        )

        return self._vectorstore, self._chunks, self._doc_metadata

    def query(
        self,
        question: str,
        chat_history: list[dict] = None,
        system_prompt: str = SYSTEM_PROMPT_DEFAULT,
    ) -> dict:
        """Execute a full RAG query through all pipeline stages.

        Args:
            question: User's question.
            chat_history: Previous chat messages.
            system_prompt: System prompt for the LLM.

        Returns:
            Dict with 'answer', 'context', 'pipeline_info' keys.
        """
        pipeline_info = {}

        # Stage 4: Query rewriting
        rewritten = rewrite_query(question, chat_history)
        expanded = expand_query(rewritten)
        pipeline_info["rewritten_query"] = rewritten
        pipeline_info["expanded_queries"] = expanded

        # Stage 5: Hybrid retrieval (semantic + BM25)
        retrieved = self._retriever.retrieve(rewritten, chat_history)
        pipeline_info["retrieved_count"] = len(retrieved)

        # Stage 6: Reranking
        reranked = self._reranker.rerank(rewritten, retrieved, top_k=self.retriever_k)
        pipeline_info["reranked_count"] = len(reranked)

        # Stage 7: Context selection (dedup + format)
        context_docs = self._select_context(reranked)
        pipeline_info["context_count"] = len(context_docs)

        # Stage 8: LLM generation
        llm = build_llm(self.model, self.temperature, self.nim_api_key)
        rag_chain = build_rag_chain_from_docs(context_docs, llm, system_prompt)
        response = rag_chain.invoke({"input": rewritten})

        return {
            "answer": response["answer"],
            "context": context_docs,
            "pipeline_info": pipeline_info,
        }

    def _select_context(self, documents: list[Document]) -> list[Document]:
        """Select and deduplicate context documents.

        Args:
            documents: Reranked list of Documents.

        Returns:
            Deduplicated list of Documents.
        """
        seen = set()
        unique = []
        for doc in documents:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(doc)
        return unique

    def get_metadata(self) -> list[DocumentMetadata]:
        """Get document metadata from processing."""
        return self._doc_metadata or []


def build_rag_chain_from_docs(
    context_docs: list[Document],
    llm,
    system_prompt: str,
):
    """Build a simple RAG chain using pre-retrieved context documents.

    This bypasses the retriever since we already have the context.

    Args:
        context_docs: Pre-retrieved context documents.
        llm: Language model instance.
        system_prompt: System prompt template.

    Returns:
        A chain that invokes the LLM with the given context.
    """
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    chain = create_stuff_documents_chain(llm, prompt_template)

    # Wrap to inject our context docs
    class ContextChain:
        def invoke(self, input_dict):
            return chain.invoke({
                "input": input_dict["input"],
                "context": context_docs,
            })

    return ContextChain()
