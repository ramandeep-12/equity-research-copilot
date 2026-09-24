import tempfile
import unittest
from pathlib import Path

import pymupdf
from test_pipeline import pdf_bytes

from source_preview import render_pdf_page


class SourcePreviewTests(unittest.TestCase):
    def test_preview_renders_pdf_and_rejects_invalid_page(self):
        with tempfile.TemporaryDirectory() as folder:
            pdf = Path(folder) / "report.pdf"
            pdf.write_bytes(pdf_bytes("Revenue 245,122    211,915    16%"))
            preview = render_pdf_page(pdf, 1)
            self.assertTrue(preview.startswith(b"\x89PNG"))
            pixmap = pymupdf.Pixmap(preview)
            self.assertLessEqual(max(pixmap.width, pixmap.height), 1801)
            for page in (0, 2):
                with self.assertRaises(ValueError):
                    render_pdf_page(pdf, page)
