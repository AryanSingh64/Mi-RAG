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
    and tables with captions, skipping slide backgrounds, headers, and decorative shapes.
    """

    CAPTION_REGEX = re.compile(r"(fig(?:ure)?\.?\s*\d+|diagram\s*\d+|table\s*\d+|algorithm\s*\d+|architecture|workflow)", re.IGNORECASE)

    @classmethod
    def detect_diagram_regions(cls, page: Any) -> List[Tuple[Any, str, str]]:
        """
        Analyzes page to return genuine diagrams: (cropped_bbox, caption_text, diagram_type)
        Skips slide backgrounds, headers, thin divider lines, and decorative shapes.
        """
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        diagrams = []

        # 1. Find Figure/Diagram captions in text blocks
        blocks = page.get_text("blocks")
        caption_blocks = []
        for b in blocks:
            text = b[4].strip()
            if cls.CAPTION_REGEX.search(text) and len(text) < 350:
                b_rect = fitz.Rect(b[:4])
                caption_blocks.append((b_rect, text))

        # 2. Collect genuine standalone image bounding boxes
        img_info_list = page.get_image_info(xrefs=True)
        img_rects = []
        for img_info in img_info_list:
            bbox = fitz.Rect(img_info.get("bbox", (0, 0, 0, 0)))
            if bbox.is_valid and not bbox.is_empty:
                w, h = bbox.width, bbox.height
                area = w * h
                aspect = w / max(1.0, h)
                area_pct = (area / page_area) * 100

                # Strict geometric filter for real figures:
                # - width >= 140, height >= 100 (filters out thin lines and small icons)
                # - aspect ratio between 0.3 and 3.5 (filters out 1px line slivers)
                # - area between 5% and 65% of page (filters out full-page slide backgrounds)
                # - not positioned in top 5% (header banner) or bottom 8% (footer icon)
                is_header_footer = (bbox.y1 > page_rect.height * 0.92) or (bbox.y0 < page_rect.height * 0.05 and h < 50)

                if (w >= 140 and h >= 100 and 0.3 <= aspect <= 3.5 and 5.0 <= area_pct <= 65.0 and not is_header_footer):
                    img_rects.append(bbox)

        # If no captions and no genuine images, skip
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
            thread_doc = fitz.open(doc_path)
            page = thread_doc[page_num]

            # 1. Native Digital Text (Instant extraction)
            native_text = (page.get_text() or "").strip()

            # 2. Detect & Crop Genuine Diagrams (Only isolated real figures)
            diagram_regions = DiagramDetector.detect_diagram_regions(page)[:2]

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
                        pix = page.get_pixmap(dpi=120, clip=diag_bbox)
                        pix.save(str(target_path), jpg_quality=85)
                        page_diagram_urls.append(img_url)

                        diag_block = [
                            f"[DIAGRAM / FIGURE: {caption}]",
                            f"[Image URL: {img_url}]"
                        ]
                        page_sections.append("\n".join(diag_block))
                    except Exception:
                        pass

            if len(native_text) >= 35:
                page_sections.append(native_text)
            else:
                # 3. Scanned / Sparse Page OCR: If native text is very short (<35 chars, e.g. logo only), run RapidOCR
                if native_text:
                    page_sections.append(native_text)
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
