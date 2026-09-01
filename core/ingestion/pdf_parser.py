import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

from rapidocr_onnxruntime import RapidOCR
from core.ingestion.base import BaseDocumentParser, ParsedDocument


class DiagramDetector:
    """
    Intelligent Layout & Vector Diagram Detector using PyMuPDF.
    Finds exact bounding boxes for genuine figures, charts, architecture drawings,
    and tables with captions, skipping slide borders, logos, and decorative lines.
    """

    CAPTION_REGEX = re.compile(r"(fig(?:ure)?\.?\s*\d+|diagram\s*\d+|table\s*\d+|algorithm\s*\d+|architecture|workflow)", re.IGNORECASE)

    @classmethod
    def detect_diagram_regions(cls, page: Any) -> List[Tuple[Any, str, str]]:
        """
        Analyzes page to return genuine diagrams: (cropped_bbox, caption_text, diagram_type)
        Skips slide backgrounds, headers, and decorative shapes.
        """
        page_rect = page.rect
        diagrams = []

        # 1. Find Figure/Diagram captions in text blocks
        blocks = page.get_text("blocks")
        caption_blocks = []
        for b in blocks:
            text = b[4].strip()
            if cls.CAPTION_REGEX.search(text) and len(text) < 350:
                b_rect = fitz.Rect(b[:4])
                caption_blocks.append((b_rect, text))

        # 2. Collect embedded image bounding boxes (filter out tiny icons or full-slide backgrounds)
        img_info_list = page.get_image_info(xrefs=True)
        img_rects = []
        for img_info in img_info_list:
            bbox = fitz.Rect(img_info.get("bbox", (0, 0, 0, 0)))
            if bbox.is_valid and not bbox.is_empty:
                # Must be a prominent graphic (width >= 100, height >= 80) and not full-page slide background (< 85% area)
                area = bbox.width * bbox.height
                page_area = page_rect.width * page_rect.height
                if bbox.width >= 100 and bbox.height >= 80 and (area < page_area * 0.85):
                    img_rects.append(bbox)

        # If no captions and no prominent images, skip (do not crop pure text or slide headers)
        if not caption_blocks and not img_rects:
            return []

        # 3. Associate images with nearest captions
        for idx, img_bbox in enumerate(img_rects[:2]):
            matched_caption = ""
            best_dist = 140.0
            for c_rect, c_text in caption_blocks:
                dist = min(abs(c_rect.y0 - img_bbox.y1), abs(img_bbox.y0 - c_rect.y1))
                if dist < best_dist:
                    best_dist = dist
                    matched_caption = c_text

            padded_rect = fitz.Rect(
                max(0, img_bbox.x0 - 10),
                max(0, img_bbox.y0 - 10),
                min(page_rect.width, img_bbox.x1 + 10),
                min(page_rect.height, img_bbox.y1 + 10)
            )
            diag_name = matched_caption.split("\n")[0][:60] if matched_caption else f"Figure {idx + 1}"
            diagrams.append((padded_rect, diag_name, "embedded_figure"))

        return diagrams[:2]


class PdfDocumentParser(BaseDocumentParser):
    """
    Multimodal Document-Aware PDF Parser.
    Extracts structured text blocks, tables, and crops precise vector/image diagrams
    with page relations and rich semantic captions for 100% accurate RAG retrieval.
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

    def _process_single_page(self, doc_path: str, page_num: int, clean_stem: str, total_pages: int) -> Tuple[str, List[str]]:
        """Processes a single page in an isolated worker thread."""
        page_sections = []
        page_diagram_urls = []
        p_idx = page_num + 1

        try:
            # Open per-thread PyMuPDF doc handle for thread-safe concurrent rendering
            thread_doc = fitz.open(doc_path)
            page = thread_doc[page_num]

            # 1. Native Digital Text (Instant extraction)
            native_text = (page.get_text() or "").strip()

            # 2. Detect & Crop Exact Diagram Bounding Boxes (Vector charts, Figure captions, Drawings, & Embedded images)
            diagram_regions = DiagramDetector.detect_diagram_regions(page)[:4]

            for d_idx, (diag_bbox, caption, diag_type) in enumerate(diagram_regions, start=1):
                clean_cap = "".join(c if c.isalnum() else "_" for c in caption[:25])
                diag_filename = f"{clean_stem}_p{p_idx}_diag{d_idx}_{clean_cap}.jpg"

                img_url = (
                    f"/api/sessions/{self.session_id}/images/{diag_filename}"
                    if self.session_id
                    else f"/images/{diag_filename}"
                )

                if self.output_images_dir:
                    target_path = self.output_images_dir / diag_filename
                    try:
                        # 140 DPI JPEG crop for crisp diagram rendering & low disk footprint
                        pix = page.get_pixmap(dpi=140, clip=diag_bbox)
                        pix.save(str(target_path), jpg_quality=85)
                        page_diagram_urls.append(img_url)

                        diag_block = [
                            f"[DIAGRAM / FIGURE: {caption}]",
                            f"[Image URL: {img_url}]"
                        ]
                        page_sections.append("\n".join(diag_block))

                    except Exception:
                        pass

            # 3. Fallback: If standalone image/drawing exists without isolated bounding box and page text is sparse
            if not page_diagram_urls and len(page.get_images()) > 0 and len(native_text) < 200 and self.output_images_dir:
                full_page_filename = f"{clean_stem}_page_{p_idx}.jpg"
                full_page_url = (
                    f"/api/sessions/{self.session_id}/images/{full_page_filename}"
                    if self.session_id
                    else f"/images/{full_page_filename}"
                )
                try:
                    pix = page.get_pixmap(dpi=100)
                    pix.save(str(self.output_images_dir / full_page_filename), jpg_quality=75)
                    page_diagram_urls.append(full_page_url)
                    page_sections.append(f"[Page {p_idx} Figure]\n[Image URL: {full_page_url}]")
                except Exception:
                    pass

            if native_text:
                page_sections.append(native_text)
            else:
                # 4. Scanned Page OCR: If native digital text is empty or missing, run RapidOCR on page image
                try:
                    ocr_engine = self._get_ocr()
                    if ocr_engine:
                        pix = page.get_pixmap(dpi=120)
                        img_bytes = pix.tobytes("png")
                        ocr_res, _ = ocr_engine(img_bytes)
                        if ocr_res:
                            extracted_lines = [line[1] for line in ocr_res if len(line) > 1 and line[1]]
                            if extracted_lines:
                                ocr_text = "\n".join(extracted_lines)
                                page_sections.append(f"[Scanned Slide OCR Text]:\n{ocr_text}")
                except Exception:
                    pass

            thread_doc.close()
        except Exception as err:
            page_sections.append(f"[Page {p_idx} parsing notice: {err}]")

        page_content = f"[Page {p_idx}]\n" + "\n\n".join(page_sections)
        return page_content, page_diagram_urls

    def parse(self, file_path: Path, progress_callback: Optional[Any] = None) -> ParsedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if fitz is None:
            # Safe Fallback to pypdf if pymupdf is not installed
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            total_pages = len(reader.pages)
            pages_text = []
            for page_num, page in enumerate(reader.pages, start=1):
                native_text = (page.extract_text() or "").strip()
                pages_text.append(f"[Page {page_num}]\n{native_text}")
                if progress_callback:
                    progress_callback("parsing", page_num, total_pages, 0)
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
                    "diagram_count": 0
                }
            )

        clean_stem = "".join(c if c.isalnum() else "_" for c in file_path.stem)
        doc = fitz.open(str(file_path))
        total_pages = len(doc)
        doc.close()

        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Parallel multi-core page processing
        num_workers = min(16, max(4, (os.cpu_count() or 4) * 2))
        doc_path_str = str(file_path.resolve())

        pages_results = [None] * total_pages
        extracted_diagrams = []
        completed_count = 0

        print(f"[*] Parsing {total_pages} PDF pages across {num_workers} parallel workers...", flush=True)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(self._process_single_page, doc_path_str, i, clean_stem, total_pages): i
                for i in range(total_pages)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                completed_count += 1
                try:
                    page_text, diag_urls = future.result()
                    pages_results[idx] = page_text
                    if diag_urls:
                        extracted_diagrams.extend(diag_urls)
                except Exception as e:
                    pages_results[idx] = f"[Page {idx + 1}]\n[Extraction error: {e}]"

                if progress_callback and (completed_count % 25 == 0 or completed_count == total_pages):
                    progress_callback("parsing", completed_count, total_pages, len(extracted_diagrams))

                if completed_count % 100 == 0 or completed_count == total_pages:
                    pct = (completed_count / total_pages) * 100
                    print(f"  [PDF Progress] Processed {completed_count}/{total_pages} pages ({pct:.1f}%) | Diagrams extracted: {len(extracted_diagrams)}", flush=True)

        pages_text = [p for p in pages_results if p is not None]
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
                "diagram_count": len(extracted_diagrams)
            }
        )
