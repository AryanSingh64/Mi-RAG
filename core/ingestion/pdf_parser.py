import io
from pathlib import Path
from typing import List, Optional
from PIL import Image
from pypdf import PdfReader
from rapidocr_onnxruntime import RapidOCR
from core.ingestion.base import BaseDocumentParser, ParsedDocument


class PdfDocumentParser(BaseDocumentParser):
    """
    Multimodal Hybrid PDF Parser.
    Extracts native digital text AND crops/extracts all embedded diagrams, formulas, charts,
    and images on every page, saving them for visual retrieval and running OCR.
    """

    def __init__(self, output_images_dir: Optional[Path] = None, session_id: Optional[str] = None):
        self.ocr_engine = None  # Lazy-load OCR engine only when images are found
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

        reader = PdfReader(str(file_path))
        pages_text = []
        has_images = False
        extracted_image_urls = []

        for page_num, page in enumerate(reader.pages, start=1):
            page_sections = []

            # 1. Extract Native Text
            native_text = (page.extract_text() or "").strip()
            if native_text:
                page_sections.append(native_text)

            # 2. Extract and crop embedded images / diagrams from page
            if len(page.images) > 0:
                ocr = self._get_ocr()
                for img_idx, img in enumerate(page.images, start=1):
                    try:
                        # Clean filename for cropped diagram
                        clean_stem = "".join(c if c.isalnum() else "_" for c in file_path.stem)
                        img_filename = f"{clean_stem}_p{page_num}_img{img_idx}.png"

                        img_url = (
                            f"/api/sessions/{self.session_id}/images/{img_filename}"
                            if self.session_id
                            else f"/images/{img_filename}"
                        )

                        # Save cropped image to session extracted_images directory
                        if self.output_images_dir:
                            target_path = self.output_images_dir / img_filename
                            # Normalize with PIL to ensure valid PNG format
                            try:
                                pil_img = Image.open(io.BytesIO(img.data)).convert("RGB")
                                pil_img.save(target_path, format="PNG")
                            except Exception:
                                with open(target_path, "wb") as f:
                                    f.write(img.data)

                        # Run OCR on image bytes
                        res, _ = ocr(img.data)
                        ocr_lines = []
                        if res:
                            ocr_lines = [item[1].strip() for item in res if item[1].strip()]

                        has_images = True
                        extracted_image_urls.append(img_url)

                        diagram_section = [
                            f"[Visual Embedded Image / Diagram: {img_filename}]",
                            f"[Image URL: {img_url}]"
                        ]
                        if ocr_lines:
                            diagram_section.append(f"[Diagram / Image Text]:\n" + "\n".join(ocr_lines))
                        else:
                            diagram_section.append("[Diagram / Image]: Visual graphic or diagram embedded on page.")

                        page_sections.append("\n".join(diagram_section))

                    except Exception as e:
                        print(f"[*] Note extracting PDF image: {e}")

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
                "has_ocr_images": has_images,
                "image_count": len(extracted_image_urls)
            }
        )
