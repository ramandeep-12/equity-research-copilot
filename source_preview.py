"""Show source tables as printed, without guessing columns from extracted text."""
from pathlib import Path

import pymupdf


def render_pdf_page(path, page_number):
    """Render a one-based citation page, with bounded image dimensions."""
    with pymupdf.open(Path(path)) as document:
        if page_number < 1 or page_number > len(document):
            raise ValueError("Source page is outside this PDF.")
        page = document[page_number - 1]
        scale = min(2.0, 1800 / max(page.rect.width, page.rect.height))
        return page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False).tobytes("png")
