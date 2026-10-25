import asyncio
import tempfile
import unittest
from pathlib import Path

import fitz
from langchain_core.documents import Document

from etl.pdf_loader import load_pdf
from etl.vision_loader import analyze_pdf_pages, enrich_pdf_documents_with_vision


class FakeVisionGateway:
    async def complete_with_image(
        self,
        system,
        user,
        image,
        *,
        mime_type,
        temperature,
        max_tokens,
    ):
        await asyncio.sleep(0.005)
        page_num = int(user.split("page ", 1)[1].split(" ", 1)[0])
        return f"Handwritten text: note for page {page_num}"


class TestVisionPageProcessing(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pdf_path = Path(self.temp_dir.name) / "notes.pdf"
        with fitz.open() as pdf:
            for page_num in range(1, 4):
                page = pdf.new_page()
                page.insert_text((72, 72), f"Printed content on page {page_num}")
            pdf.save(self.pdf_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_analyzes_every_pdf_page(self):
        descriptions = asyncio.run(
            analyze_pdf_pages(
                str(self.pdf_path),
                gateway=FakeVisionGateway(),
            )
        )

        self.assertEqual(set(descriptions), {1, 2, 3})
        self.assertIn("note for page 3", descriptions[3])

    def test_vision_text_is_added_to_searchable_page_documents(self):
        documents = load_pdf(str(self.pdf_path), include_empty_pages=True)
        enriched = enrich_pdf_documents_with_vision(
            documents,
            gateway=FakeVisionGateway(),
        )

        self.assertEqual(len(enriched), 3)
        self.assertIn("Printed content on page 2", enriched[1].page_content)
        self.assertIn("note for page 2", enriched[1].page_content)
        self.assertTrue(enriched[1].metadata["vision_analyzed"])

    def test_vision_loader_keeps_scanned_pages_for_indexing(self):
        scanned_path = Path(self.temp_dir.name) / "scanned.pdf"
        with fitz.open() as pdf:
            pdf.new_page()
            pdf.new_page()
            pdf.save(scanned_path)

        documents = load_pdf(str(scanned_path), include_empty_pages=True)

        self.assertEqual(len(documents), 2)
        self.assertEqual([doc.metadata["page"] for doc in documents], [1, 2])
        self.assertTrue(all(not doc.page_content for doc in documents))


if __name__ == "__main__":
    unittest.main()
