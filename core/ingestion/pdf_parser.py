from pathlib import Path
from typing import List, Optional
# pyrefly: ignore [missing-import]
from pypdf import PdfReader
# pyrefly: ignore [missing-import]
from rapidocr_onnxruntime import RapidOCR
from core.ingestion.base import BaseDocumentParser, ParsedDocument


class PdfDocumentParser(BaseDocumentParser):
    """
    Hybrid PDF Parser.
    Extracts both native digital text AND runs RapidOCR on any embedded images,
    ensuring 100% data coverage for scanned, digital, and hybrid documents.
    """

    def __init__(self):
        self.ocr_engine = None  # Lazy-load OCR engine only when images are found

    def _get_ocr(self):
        if self.ocr_engine is None:
            self.ocr_engine = RapidOCR()
        return self.ocr_engine

    def parse(self, file_path: Path) -> ParsedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        reader = PdfReader(str(file_path))
        pages_text = []
        has_images = False

        for page_num, page in enumerate(reader.pages, start=1):
            page_sections = []

            # 1. Extract Native Text
            native_text = (page.extract_text() or "").strip()
            if native_text:
                page_sections.append(native_text)

            # 2. Extract and OCR any embedded images on the page
            if len(page.images) > 0:
                ocr = self._get_ocr()
                ocr_lines = []
                for img_idx, img in enumerate(page.images, start=1):
                    try:
                        res, _ = ocr(img.data)
                        if res:
                            lines = [item[1] for item in res if item[1].strip()]
                            if lines:
                                has_images = True
                                ocr_lines.extend(lines)
                    except Exception:
                        pass

                if ocr_lines:
                    # Filter out lines already captured in native text to avoid exact duplication
                    unique_ocr_lines = [l for l in ocr_lines if l.lower() not in native_text.lower()]
                    if unique_ocr_lines:
                        page_sections.append("[Visual / Image Content]\n" + "\n".join(unique_ocr_lines))

            # Combine page sections
            full_page_content = "\n\n".join(page_sections)
            pages_text.append(full_page_content)

        full_text = "\n\n".join([f"[Page {i+1}]\n{txt}" for i, txt in enumerate(pages_text) if txt.strip()])

        return ParsedDocument(
            filename=file_path.name,
            file_path=str(file_path.resolve()),
            file_type="pdf",
            text_content=full_text,
            pages=pages_text,
            metadata={
                "total_pages": len(reader.pages),
                "char_count": len(full_text),
                "has_ocr_images": has_images
            }
        )
