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
    Finds exact bounding boxes for vector graphics, figures, charts, tables,
    and drawings + their surrounding captions (Fig 1, Figure 2, Diagram, etc.),
    and crops ONLY the exact visual diagram rather than the entire page.
    """

    CAPTION_REGEX = re.compile(r"(fig(?:ure)?\.?\s*\d+|diagram\s*\d+|table\s*\d+|algorithm\s*\d+|architecture)", re.IGNORECASE)

    @classmethod
    def detect_diagram_regions(cls, page: Any) -> List[Tuple[Any, str, str]]:
        """
        Analyzes page drawings, images, and text blocks to return a list of:
        (cropped_bbox, caption_text, diagram_type)
        """
        page_rect = page.rect
        diagrams = []

        # 1. Collect all vector drawings bounding boxes
        drawings = page.get_drawings()
        drawing_rects = []
        for d in drawings:
            r = d.get("rect")
            if r and r.is_valid and not r.is_empty:
                # Filter out full-page borders or tiny decoration dots
                if r.width > 20 and r.height > 20 and (r.width < page_rect.width * 0.95 or r.height < page_rect.height * 0.95):
                    drawing_rects.append(r)

        # 2. Collect embedded image bounding boxes
        img_info_list = page.get_image_info(xrefs=True)
        img_rects = []
        for img_info in img_info_list:
            bbox = fitz.Rect(img_info.get("bbox", (0, 0, 0, 0)))
            if bbox.is_valid and not bbox.is_empty and bbox.width > 30 and bbox.height > 30:
                img_rects.append(bbox)

        # 3. Find Figure/Diagram captions in text blocks
        blocks = page.get_text("blocks")
        caption_blocks = []
        for b in blocks:
            text = b[4].strip()
            if cls.CAPTION_REGEX.search(text) and len(text) < 350:
                b_rect = fitz.Rect(b[:4])
                caption_blocks.append((b_rect, text))

        # 4. Group drawings and images into clustered diagram areas
        all_visual_rects = drawing_rects + img_rects
        clustered_regions = []

        for v_rect in all_visual_rects:
            merged = False
            for i, c_rect in enumerate(clustered_regions):
                # If visual items are close to each other (within 35pt), merge their bounding box
                expanded = fitz.Rect(c_rect).include_point((v_rect.x0 - 25, v_rect.y0 - 25))
                expanded.include_point((v_rect.x1 + 25, v_rect.y1 + 25))
                if expanded.intersects(v_rect) or c_rect.intersects(v_rect):
                    clustered_regions[i] = c_rect | v_rect
                    merged = True
                    break
            if not merged:
                clustered_regions.append(v_rect)

        # 5. Associate each diagram cluster with its nearest caption
        used_captions = set()
        for idx, d_rect in enumerate(clustered_regions):
            if d_rect.width < 50 or d_rect.height < 40:
                continue

            matched_caption = ""
            best_dist = 150.0  # Max search distance for caption

            for c_idx, (c_rect, c_text) in enumerate(caption_blocks):
                # Distance between diagram bottom/top and caption
                dist = min(
                    abs(c_rect.y0 - d_rect.y1),  # Caption below diagram
                    abs(d_rect.y0 - c_rect.y1),  # Caption above diagram
                    abs(c_rect.y0 - d_rect.y0)
                )
                if dist < best_dist:
                    best_dist = dist
                    matched_caption = c_text
                    used_captions.add(c_idx)
                    # Include caption in diagram crop for complete context
                    d_rect = d_rect | c_rect

            # Add padding around diagram
            padded_rect = fitz.Rect(
                max(0, d_rect.x0 - 15),
                max(0, d_rect.y0 - 15),
                min(page_rect.width, d_rect.x1 + 15),
                min(page_rect.height, d_rect.y1 + 15)
            )

            diag_name = matched_caption.split("\n")[0][:60] if matched_caption else f"Diagram {idx + 1}"
            diagrams.append((padded_rect, diag_name, "vector_diagram" if drawing_rects else "embedded_figure"))

        # 6. Check for standalone captions without detected vector objects (e.g. text-only tables or subtle line drawings)
        for c_idx, (c_rect, c_text) in enumerate(caption_blocks):
            if c_idx not in used_captions:
                # Expand region above or below caption
                box_above = fitz.Rect(
                    max(0, c_rect.x0 - 40),
                    max(0, c_rect.y0 - 220),
                    min(page_rect.width, c_rect.x1 + 40),
                    min(page_rect.height, c_rect.y1 + 15)
                )
                diagrams.append((box_above, c_text.split("\n")[0][:60], "caption_region"))

        return diagrams


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

            # 2. Check if page has visual content (images or distinct drawings)
            image_list = page.get_images()
            has_images = len(image_list) > 0

            # Only run heavy diagram clustering if visual items exist on page
            if has_images:
                diagram_regions = DiagramDetector.detect_diagram_regions(page)[:3]

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
                            # 120 DPI JPEG crop for fast encoding & low disk footprint
                            pix = page.get_pixmap(dpi=120, clip=diag_bbox)
                            pix.save(str(target_path), jpg_quality=80)
                            page_diagram_urls.append(img_url)

                            diag_block = [
                                f"[DIAGRAM / FIGURE: {caption}]",
                                f"[Image URL: {img_url}]"
                            ]
                            page_sections.append("\n".join(diag_block))

                        except Exception:
                            pass

                # If standalone image exists without isolated bounding box and page text is sparse
                if not page_diagram_urls and len(native_text) < 150 and self.output_images_dir:
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

            thread_doc.close()
        except Exception as err:
            page_sections.append(f"[Page {p_idx} parsing notice: {err}]")

        page_content = f"[Page {p_idx}]\n" + "\n\n".join(page_sections)
        return page_content, page_diagram_urls

    def parse(self, file_path: Path) -> ParsedDocument:
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
        from concurrent.futures import ThreadPoolExecutor

        # Parallel multi-core page processing
        num_workers = min(12, max(2, (os.cpu_count() or 4) * 2))
        doc_path_str = str(file_path.resolve())

        pages_results = [None] * total_pages
        extracted_diagrams = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(self._process_single_page, doc_path_str, i, clean_stem, total_pages): i
                for i in range(total_pages)
            }
            for future in future_to_idx:
                idx = future_to_idx[future]
                try:
                    page_text, diag_urls = future.result()
                    pages_results[idx] = page_text
                    if diag_urls:
                        extracted_diagrams.extend(diag_urls)
                except Exception as e:
                    pages_results[idx] = f"[Page {idx + 1}]\n[Extraction error: {e}]"

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
