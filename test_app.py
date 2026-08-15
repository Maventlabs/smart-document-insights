import sys
import unittest

class TestAppImports(unittest.TestCase):
    def test_imports(self):
        try:
            import app
            self.assertTrue(hasattr(app, 'main'))
            self.assertTrue(hasattr(app, 'process_document'))
            print("Successfully imported app.py and verified core functions exist.")
        except ImportError as e:
            self.fail(f"Import failed: {e}")
        except Exception as e:
            self.fail(f"Unexpected error during import: {e}")

if __name__ == '__main__':
    unittest.main()
