import io
from pathlib import Path
from typing import List, Optional
from PIL import Image
from core.ingestion.base import BaseDocumentParser, ParsedDocument

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

from rapidocr_onnxruntime import RapidOCR


class PdfDocumentParser(BaseDocumentParser):
    """
    State-of-the-Art Multimodal PDF Parser powered by PyMuPDF + RapidOCR.
    Extracts native digital text, renders every page diagram/figure into crisp visual images,
    and indexes visual content with 1-click interactive image snippet retrieval.
    """

    def __init__(self, output_images_dir: Optional[Path] = None, session_id: Optional[str] = None):
        self.ocr_engine = None
        self.output_images_dir = Path(output_images_dir) if output_images_dir else None
        if self.output_images_dir:
            self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id

    def _get_ocr(self):
        if self.ocr_engine is None:
            self.ocr_engine = RapidOCR()
        return self.ocr_engine

    def parse(self, file_path: Path) -> ParsedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        clean_stem = "".join(c if c.isalnum() else "_" for c in file_path.stem)
        pages_text = []
        has_images = False
        extracted_image_urls = []

        if fitz is not None:
            # High-performance PyMuPDF Engine
            doc = fitz.open(str(file_path))
            total_pages = len(doc)

            for page_num, page in enumerate(doc, start=1):
                page_sections = []

                # 1. Extract digital text
                native_text = (page.get_text() or "").strip()

                # 2. Render page as visual diagram / document image
                page_img_filename = f"{clean_stem}_page_{page_num}.png"
                img_url = (
                    f"/api/sessions/{self.session_id}/images/{page_img_filename}"
                    if self.session_id
                    else f"/images/{page_img_filename}"
                )

                if self.output_images_dir:
                    target_path = self.output_images_dir / page_img_filename
                    try:
                        pix = page.get_pixmap(dpi=150)
                        pix.save(str(target_path))
                        extracted_image_urls.append(img_url)
                        has_images = True
                    except Exception as e:
                        print(f"[*] Note rendering PDF page image: {e}")

                # 3. If native text is very short (scanned PDF or chart page), run RapidOCR on page pixmap
                ocr_text = ""
                if len(native_text) < 100 and self.output_images_dir and (self.output_images_dir / page_img_filename).exists():
                    ocr = self._get_ocr()
                    try:
                        res, _ = ocr(str(self.output_images_dir / page_img_filename))
                        if res:
                            lines = [item[1].strip() for item in res if item[1].strip()]
                            ocr_text = "\n".join(lines)
                    except Exception:
                        pass

                # Assemble page content with Image URL anchor
                page_header = (
                    f"[Page {page_num} Document & Visual Diagram: {page_img_filename}]\n"
                    f"[Image URL: {img_url}]"
                )
                page_sections.append(page_header)

                if native_text:
                    page_sections.append(native_text)

                if ocr_text and ocr_text.lower() not in native_text.lower():
                    page_sections.append(f"[OCR Extracted Visual Text]:\n{ocr_text}")

                full_page_content = "\n\n".join(page_sections)
                pages_text.append(full_page_content)

            doc.close()
        else:
            # Fallback to pypdf
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)

            for page_num, page in enumerate(reader.pages, start=1):
                native_text = (page.extract_text() or "").strip()
                pages_text.append(f"[Page {page_num}]\n{native_text}")

        full_text = "\n\n".join(pages_text)

        return ParsedDocument(
            filename=file_path.name,
            file_path=str(file_path.resolve()),
            file_type="pdf",
            text_content=full_text,
            pages=pages_text,
            metadata={
                "total_pages": total_pages,
                "char_count": len(full_text),
                "has_ocr_images": has_images,
                "image_count": len(extracted_image_urls)
            }
        )
