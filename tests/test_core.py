"""Tests for core business logic."""

import sys
import os
import tempfile
import unittest
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestDocumentModule(unittest.TestCase):
    """Test document loading and stats."""

    def test_load_txt_file(self):
        from smart_doc.core.document import load_documents

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
            f.write("Ini adalah test file.\nBaris kedua.")
            path = f.name

        try:
            docs = load_documents([path])
            self.assertGreater(len(docs), 0)
            self.assertIn("test file", docs[0].page_content)
        finally:
            os.remove(path)

    def test_load_unsupported_file(self):
        from smart_doc.core.document import load_documents

        with self.assertRaises(ValueError) as ctx:
            load_documents(["file.xyz"])
        self.assertIn("tidak didukung", str(ctx.exception))

    def test_get_doc_stats(self):
        from smart_doc.core.document import get_doc_stats

        doc1 = MagicMock()
        doc1.page_content = "Hello world test document"
        doc2 = MagicMock()
        doc2.page_content = "Another document with more words here"

        stats = get_doc_stats([doc1, doc2])
        self.assertEqual(stats["pages"], 2)
        self.assertEqual(stats["words"], 10)
        self.assertGreater(stats["chars"], 0)

    def test_get_doc_stats_empty(self):
        from smart_doc.core.document import get_doc_stats

        stats = get_doc_stats([])
        self.assertEqual(stats["pages"], 0)
        self.assertEqual(stats["words"], 0)


class TestRAGModule(unittest.TestCase):
    """Test RAG chain building."""

    def test_build_rag_chain(self):
        from smart_doc.core.rag import build_rag_chain

        mock_vectorstore = MagicMock()
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever

        with self.mock_module("smart_doc.core.rag.create_stuff_documents_chain") as mock_stuff, \
             self.mock_module("smart_doc.core.rag.create_retrieval_chain") as mock_retrieval, \
             self.mock_module("smart_doc.core.rag.ChatPromptTemplate") as mock_prompt:

            chain = build_rag_chain(mock_vectorstore, mock_llm, 5, "System: {context}")
            mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
            mock_retrieval.assert_called_once()
            self.assertIsNotNone(chain)

    class mock_module:
        """Context manager for mocking module-level imports."""
        def __init__(self, path):
            self.path = path
            self.mock = MagicMock()

        def __enter__(self):
            parts = self.path.rsplit(".", 1)
            import sys
            parent = __import__(parts[0], fromlist=[parts[1]])
            setattr(parent, parts[1], self.mock)
            return self.mock

        def __exit__(self, *args):
            pass


class TestPrompts(unittest.TestCase):
    """Test system prompts."""

    def test_prompts_exist(self):
        from smart_doc.core.prompts import (
            SYSTEM_PROMPT_DEFAULT,
            SYSTEM_PROMPT_SUMMARY,
            SYSTEM_PROMPT_INSIGHTS,
        )
        self.assertIn("Konteks", SYSTEM_PROMPT_DEFAULT)
        self.assertIn("Konteks", SYSTEM_PROMPT_SUMMARY)
        self.assertIn("Konteks", SYSTEM_PROMPT_INSIGHTS)


if __name__ == "__main__":
    unittest.main()
