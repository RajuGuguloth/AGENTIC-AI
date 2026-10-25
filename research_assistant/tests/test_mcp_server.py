import asyncio
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp_server import build_index_from_paths


class TestMcpIndexingInputs(unittest.TestCase):
    def test_empty_paths_raise_an_error(self):
        with self.assertRaisesRegex(ValueError, "At least one PDF path"):
            asyncio.run(build_index_from_paths([]))

    def test_missing_paths_raise_an_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "PDF paths not found"):
            asyncio.run(build_index_from_paths(["/missing/paper.pdf"]))


if __name__ == "__main__":
    unittest.main()
