"""Tests for utility functions."""

import sys
import os
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestFileUtils(unittest.TestCase):
    """Test file utility functions."""

    def test_cleanup_file_existing(self):
        from smart_doc.utils.file import cleanup_file

        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"test")
            path = f.name

        self.assertTrue(os.path.exists(path))
        cleanup_file(path)
        self.assertFalse(os.path.exists(path))

    def test_cleanup_file_nonexistent(self):
        from smart_doc.utils.file import cleanup_file
        cleanup_file("/nonexistent/path/to/file.txt")  # Should not raise

    def test_cleanup_file_none(self):
        from smart_doc.utils.file import cleanup_file
        cleanup_file(None)  # Should not raise

    def test_validate_file_types(self):
        from smart_doc.utils.file import validate_file_types

        mock_files = [
            type("F", (), {"name": "doc.pdf"})(),
            type("F", (), {"name": "data.txt"})(),
            type("F", (), {"name": "image.exe"})(),
        ]
        invalid = validate_file_types(mock_files, ["pdf", "txt", "docx"])
        self.assertEqual(len(invalid), 1)
        self.assertEqual(invalid[0].name, "image.exe")


class TestExportUtils(unittest.TestCase):
    """Test export functions."""

    def test_export_chat_empty(self):
        from smart_doc.utils.export import export_chat_to_markdown

        result = export_chat_to_markdown([])
        self.assertIn("Chat History", result)

    def test_export_chat_with_messages(self):
        from smart_doc.utils.export import export_chat_to_markdown

        messages = [
            {"role": "user", "content": "Apa isi dokumen ini?"},
            {"role": "assistant", "content": "Dokumen ini berisi laporan keuangan."},
        ]
        result = export_chat_to_markdown(messages)
        self.assertIn("Pengguna", result)
        self.assertIn("Asisten", result)

    def test_export_to_markdown(self):
        from smart_doc.utils.export import export_to_markdown

        result = export_to_markdown("Test Title", "Some content here")
        self.assertIn("Test Title", result)
        self.assertIn("Some content here", result)


if __name__ == "__main__":
    unittest.main()
