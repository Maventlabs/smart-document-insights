"""Tests for module imports and configuration."""

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestImports(unittest.TestCase):
    """Test that all modules can be imported."""

    def test_import_config(self):
        from smart_doc import config
        self.assertTrue(hasattr(config, "SUPPORTED_FILE_TYPES"))
        self.assertTrue(hasattr(config, "NIM_BASE_URL"))
        self.assertTrue(hasattr(config, "NIM_CHAT_MODELS"))
        self.assertEqual(config.NIM_BASE_URL, "https://integrate.api.nvidia.com/v1")

    def test_import_core(self):
        from smart_doc.core import (
            load_documents,
            process_documents,
            build_rag_chain,
            invoke_rag,
            SYSTEM_PROMPT_DEFAULT,
            SYSTEM_PROMPT_SUMMARY,
            SYSTEM_PROMPT_INSIGHTS,
        )
        self.assertTrue(callable(load_documents))
        self.assertTrue(callable(process_documents))
        self.assertTrue(callable(build_rag_chain))
        self.assertTrue(callable(invoke_rag))

    def test_import_ui(self):
        from smart_doc.ui.sidebar import render_sidebar
        from smart_doc.ui.stats import render_document_stats
        from smart_doc.ui.chat import render_chat_tab
        from smart_doc.ui.summary import render_summary_tab
        from smart_doc.ui.insights import render_insights_tab
        self.assertTrue(callable(render_sidebar))
        self.assertTrue(callable(render_document_stats))
        self.assertTrue(callable(render_chat_tab))
        self.assertTrue(callable(render_summary_tab))
        self.assertTrue(callable(render_insights_tab))

    def test_import_utils(self):
        from smart_doc.utils.file import save_uploaded_file, cleanup_file, validate_file_types
        from smart_doc.utils.export import export_chat_to_markdown, export_to_markdown
        self.assertTrue(callable(save_uploaded_file))
        self.assertTrue(callable(cleanup_file))
        self.assertTrue(callable(validate_file_types))
        self.assertTrue(callable(export_chat_to_markdown))
        self.assertTrue(callable(export_to_markdown))


class TestConfig(unittest.TestCase):
    """Test configuration values."""

    def test_nim_constants(self):
        from smart_doc import config
        self.assertIn("nvidia", config.NIM_BASE_URL)
        self.assertIn("embed", config.NIM_EMBEDDING_MODEL.lower())
        self.assertGreater(len(config.NIM_CHAT_MODELS), 0)

    def test_supported_types(self):
        from smart_doc import config
        self.assertIn("pdf", config.SUPPORTED_FILE_TYPES)
        self.assertIn("txt", config.SUPPORTED_FILE_TYPES)
        self.assertIn("docx", config.SUPPORTED_FILE_TYPES)

    def test_rag_defaults(self):
        from smart_doc import config
        self.assertEqual(config.CHUNK_SIZE_DEFAULT, 1000)
        self.assertEqual(config.CHUNK_OVERLAP_DEFAULT, 200)
        self.assertEqual(config.RETRIEVER_K_DEFAULT, 5)


if __name__ == "__main__":
    unittest.main()
