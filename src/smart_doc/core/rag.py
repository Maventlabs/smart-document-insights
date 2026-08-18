"""RAG (Retrieval-Augmented Generation) chain building and invocation."""

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from smart_doc.config import NIM_BASE_URL


def build_llm(model: str, temperature: float, api_key: str) -> ChatOpenAI:
    """Build an LLM instance connected to NVIDIA NIM.

    Args:
        model: Model identifier (e.g. 'meta/llama-3.1-8b-instruct').
        temperature: Sampling temperature (0.0 - 1.0).
        api_key: NVIDIA NIM API key.

    Returns:
        Configured ChatOpenAI instance.
    """
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=NIM_BASE_URL,
        api_key=api_key,
    )


def build_rag_chain(vectorstore, llm, retriever_k: int, system_prompt: str):
    """Build a RAG (Retrieval-Augmented Generation) chain.

    Args:
        vectorstore: Chroma vector store instance.
        llm: Language model instance.
        retriever_k: Number of chunks to retrieve.
        system_prompt: System prompt template with {context} placeholder.

    Returns:
        Configured RAG chain.
    """
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


def invoke_rag(rag_chain, question: str) -> dict:
    """Invoke the RAG chain and return the response dict.

    Args:
        rag_chain: The RAG chain to invoke.
        question: The user's question.

    Returns:
        Dict with 'answer' and 'context' keys.
    """
    return rag_chain.invoke({"input": question})
