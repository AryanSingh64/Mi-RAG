import threading
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
    Intelligent Layout & Multimodal Diagram Detector using PyMuPDF.
    Finds exact bounding boxes for genuine figures, charts, architecture drawings,
    vector plots, and tables with captions, skipping slide backgrounds, headers, and decorative shapes.
    """

    CAPTION_REGEX = re.compile(
        r"^\s*(fig(?:ure)?\.?\s*\d+|table\s*\d+|chart\s*\d+|diagram\s*\d+|algorithm\s*\d+|scheme\s*\d+|plate\s*\d+|photo\s*\d+|map\s*\d+|workflow|architecture)",
        re.IGNORECASE
    )

    @classmethod
    def detect_diagram_regions(cls, page: Any) -> List[Tuple[Any, str, str]]:
        """
        Analyzes page to return complete publication-grade figures, diagrams, and tables:
        (cropped_bbox, caption_text, diagram_type).
        Reconstructs compound figures (vector drawings + embedded sub-images + callouts)
        into their full, unfragmented bounding boxes.
        """
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        native_page_text = (page.get_text() or "").strip()

        # 1. Find all Figure/Diagram/Table captions in text blocks
        blocks = page.get_text("blocks")
        caption_blocks = []
        for b in blocks:
            text = b[4].strip()
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if lines and cls.CAPTION_REGEX.search(lines[0]) and len(text) < 500:
                caption_blocks.append((fitz.Rect(b[:4]), lines[0], text))

        drawings = page.get_drawings()
        img_info = page.get_image_info(xrefs=True)

        diagrams = []
        covered_rects = []

        # 2. First Pass: Reconstruct Complete Figures & Diagrams around detected Captions
        for c_rect, first_line, full_caption in caption_blocks:
            # Check elements ABOVE caption (Standard Figure / Chart / Diagram / Scheme / Flowchart)
            above_drawings = [d['rect'] for d in drawings if d['rect'].y1 <= c_rect.y0 + 12 and d['rect'].y0 >= c_rect.y0 - 580]
            above_images = [fitz.Rect(img['bbox']) for img in img_info if fitz.Rect(img['bbox']).y1 <= c_rect.y0 + 12 and fitz.Rect(img['bbox']).y0 >= c_rect.y0 - 580]

            all_above = [r for r in (above_drawings + above_images) if r.width > 5 and r.height > 5]
            if all_above:
                combined = fitz.Rect(all_above[0])
                for r in all_above[1:]:
                    combined |= r

                if combined.width >= 50 and combined.height >= 35:
                    full_fig_bbox = fitz.Rect(
                        max(0, min(combined.x0, c_rect.x0) - 6),
                        max(0, combined.y0 - 6),
                        min(page_rect.width, max(combined.x1, c_rect.x1) + 6),
                        min(page_rect.height, c_rect.y1 + 6)
                    )
                    cap_clean = first_line.split("\n")[0][:80]
                    diagrams.append((full_fig_bbox, cap_clean, "figure"))
                    covered_rects.append(full_fig_bbox)
                    continue

            # Check elements BELOW caption (Standard Table / Algorithm / Top Title)
            below_drawings = [d['rect'] for d in drawings if d['rect'].y0 >= c_rect.y1 - 12 and d['rect'].y1 <= c_rect.y1 + 580]
            below_images = [fitz.Rect(img['bbox']) for img in img_info if fitz.Rect(img['bbox']).y0 >= c_rect.y1 - 12 and fitz.Rect(img['bbox']).y1 <= c_rect.y1 + 580]

            all_below = [r for r in (below_drawings + below_images) if r.width > 5 and r.height > 5]
            if all_below:
                combined = fitz.Rect(all_below[0])
                for r in all_below[1:]:
                    combined |= r

                if combined.width >= 50 and combined.height >= 35:
                    full_fig_bbox = fitz.Rect(
                        max(0, min(combined.x0, c_rect.x0) - 6),
                        max(0, c_rect.y0 - 6),
                        min(page_rect.width, max(combined.x1, c_rect.x1) + 6),
                        min(page_rect.height, combined.y1 + 6)
                    )
                    cap_clean = first_line.split("\n")[0][:80]
                    diagrams.append((full_fig_bbox, cap_clean, "table"))
                    covered_rects.append(full_fig_bbox)
                    continue

        # 3. Second Pass: Genuine standalone raster images (Photos, standalone graphics)
        # Skip sub-images that are ALREADY inside a covered figure region
        for idx, img in enumerate(img_info, start=1):
            bbox = fitz.Rect(img.get('bbox', (0, 0, 0, 0)))
            if not bbox.is_valid or bbox.is_empty:
                continue

            w, h = bbox.width, bbox.height
            area = w * h
            if w < 50 or h < 50 or area < 4000:
                continue

            # Check if this raster image is inside an already detected figure
            is_inside_figure = False
            for cr in covered_rects:
                intersection = bbox & cr
                if intersection.is_valid and not intersection.is_empty:
                    if (intersection.width * intersection.height) / area > 0.35:
                        is_inside_figure = True
                        break

            if is_inside_figure:
                continue

            # Standalone genuine image
            aspect = w / max(1.0, h)
            if aspect > 20.0 or aspect < 0.05:
                continue
            if (area / page_area) > 0.90:
                continue

            padded_rect = fitz.Rect(
                max(0, bbox.x0 - 4),
                max(0, bbox.y0 - 4),
                min(page_rect.width, bbox.x1 + 4),
                min(page_rect.height, bbox.y1 + 4)
            )
            diagrams.append((padded_rect, f"Figure (Page Image {idx})", "raster_photo"))
            covered_rects.append(padded_rect)

        return diagrams


class PdfDocumentParser(BaseDocumentParser):
    """
    Multimodal Document-Aware PDF Parser.
    Extracts structured text blocks, tables, and crops precise vector/image diagrams
    with page relations and rich semantic captions for 100% accurate RAG retrieval.
    """

    def __init__(
        self,
        output_images_dir: Optional[Path] = None,
        session_id: Optional[str] = None,
        vision_parser: Optional[Any] = None
    ):
        self.ocr_engine = None
        self._ocr_lock = threading.Lock()
        self.output_images_dir = Path(output_images_dir) if output_images_dir else None
        if self.output_images_dir:
            self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.vision_parser = vision_parser

    def _get_ocr(self):
        if self.ocr_engine is None:
            with self._ocr_lock:
                if self.ocr_engine is None:
                    self.ocr_engine = RapidOCR()
        return self.ocr_engine

    def _process_single_page(self, doc_path: str, page_num: int, clean_stem: str, total_pages: int) -> Tuple[str, List[str]]:
        """Processes a single page in an isolated worker thread."""
        page_sections = []
        page_image_urls = []
        p_idx = page_num + 1

        try:
            thread_doc = fitz.open(doc_path)
            page = thread_doc[page_num]

            # 1. Native Digital Text (Instant extraction)
            native_text = (page.get_text() or "").strip()
            is_visual_doc_page = len(native_text) < 120

            # 2. Detect & Crop Genuine Images/Figures (All isolated real figures and vector charts)
            image_regions = DiagramDetector.detect_diagram_regions(page)

            for d_idx, (diag_bbox, caption, diag_type) in enumerate(image_regions, start=1):
                clean_cap = "".join(c if c.isalnum() else "_" for c in caption[:25])
                img_filename = f"{clean_stem}_p{p_idx}_img{d_idx}_{clean_cap}.jpg"

                img_url = (
                    f"/api/sessions/{self.session_id}/images/{img_filename}"
                    if self.session_id
                    else f"/images/{img_filename}"
                )

                if self.output_images_dir:
                    target_path = self.output_images_dir / img_filename
                    try:
                        pix = page.get_pixmap(dpi=140, clip=diag_bbox)
                        pix.save(str(target_path), jpg_quality=90)
                        page_image_urls.append(img_url)

                        img_block = [
                            f"[IMAGE / FIGURE: {caption}]",
                            f"[Image URL: {img_url}]"
                        ]

                        # Deep vision analysis for visual-dominant / true-image pages
                        if is_visual_doc_page and self.vision_parser and hasattr(self.vision_parser, "describe_and_ocr_image"):
                            try:
                                vis_res = self.vision_parser.describe_and_ocr_image(target_path)
                                vis_desc = vis_res.get("description", "").strip()
                                if vis_desc:
                                    img_block.append(f"[Visual Scene, Composition & Details Analysis]:\n{vis_desc}")
                            except Exception:
                                pass

                        page_sections.append("\n".join(img_block))
                    except Exception:
                        pass

            # 3. Direct Image XObject Extraction for uncaptured standalone graphics
            if not page_image_urls:
                try:
                    for img_tuple in page.get_images(full=True):
                        xref = img_tuple[0]
                        img_dict = thread_doc.extract_image(xref)
                        if img_dict:
                            w, h = img_dict.get("width", 0), img_dict.get("height", 0)
                            if w >= 80 and h >= 80 and (w * h >= 8000):
                                ext = img_dict.get("ext", "jpg")
                                raw_filename = f"{clean_stem}_p{p_idx}_raw_xref{xref}.{ext}"
                                raw_url = f"/api/sessions/{self.session_id}/images/{raw_filename}" if self.session_id else f"/images/{raw_filename}"
                                if raw_url not in page_image_urls and self.output_images_dir:
                                    raw_target = self.output_images_dir / raw_filename
                                    if not raw_target.exists():
                                        raw_target.write_bytes(img_dict["image"])
                                    page_image_urls.append(raw_url)
                                    page_sections.append(f"[IMAGE / FIGURE: Embedded Graphic {xref} (Page {p_idx})]\n[Image URL: {raw_url}]")
                except Exception:
                    pass

            # 4. Fallback for scanned slides, catalogs, posters, and image-heavy pages
            if not page_image_urls and len(native_text) < 60:
                visual_filename = f"{clean_stem}_p{p_idx}_visual.jpg"
                img_url = (
                    f"/api/sessions/{self.session_id}/images/{visual_filename}"
                    if self.session_id
                    else f"/images/{visual_filename}"
                )
                if self.output_images_dir:
                    target_path = self.output_images_dir / visual_filename
                    try:
                        pix = page.get_pixmap(dpi=140)
                        pix.save(str(target_path), jpg_quality=90)
                        page_image_urls.append(img_url)
                        
                        full_img_block = [
                            f"[IMAGE / FIGURE: Page {p_idx} Visual Document / Photo]",
                            f"[Image URL: {img_url}]"
                        ]

                        # Multimodal Vision Model scene description for image-only/sparse pages
                        if self.vision_parser and hasattr(self.vision_parser, "describe_and_ocr_image"):
                            try:
                                vis_res = self.vision_parser.describe_and_ocr_image(target_path)
                                vis_desc = vis_res.get("description", "").strip()
                                if vis_desc:
                                    full_img_block.append(f"[Visual Scene, Composition & Details Analysis of Page {p_idx}]:\n{vis_desc}")
                            except Exception:
                                pass

                        page_sections.append("\n".join(full_img_block))
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
                        pix = page.get_pixmap(dpi=140)
                        img_bytes = pix.tobytes("png")
                        with self._ocr_lock:
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
        return page_content, page_image_urls

    def parse(
        self,
        file_path: Path,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        progress_callback: Optional[Any] = None
    ) -> ParsedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if fitz is None:
            # Safe Fallback to pypdf if pymupdf is not installed
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            total_doc_pages = len(reader.pages)
            s_idx = max(0, (start_page - 1)) if start_page is not None else 0
            e_idx = min(total_doc_pages, end_page) if end_page is not None else total_doc_pages
            if s_idx >= e_idx:
                s_idx, e_idx = 0, total_doc_pages

            pages_text = []
            for i, p_num in enumerate(range(s_idx, e_idx), start=1):
                page = reader.pages[p_num]
                native_text = (page.extract_text() or "").strip()
                pages_text.append(f"[Page {p_num + 1}]\n{native_text}")
                if progress_callback:
                    progress_callback("parsing", i, (e_idx - s_idx), 0)
            full_text = "\n\n".join(pages_text)
            return ParsedDocument(
                filename=file_path.name,
                file_path=str(file_path.resolve()),
                file_type="pdf",
                text_content=full_text,
                pages=pages_text,
                metadata={
                    "total_pages": (e_idx - s_idx),
                    "total_doc_pages": total_doc_pages,
                    "page_range": f"{s_idx + 1}-{e_idx}",
                    "char_count": len(full_text),
                    "extracted_images": [],
                    "diagram_count": 0
                }
            )

        clean_stem = "".join(c if c.isalnum() else "_" for c in file_path.stem)
        doc = fitz.open(str(file_path))
        total_doc_pages = len(doc)
        doc.close()

        # Handle 2-way page range slicing
        s_idx = max(0, (start_page - 1)) if start_page is not None else 0
        e_idx = min(total_doc_pages, end_page) if end_page is not None else total_doc_pages
        if s_idx >= e_idx:
            s_idx, e_idx = 0, total_doc_pages
        
        page_indices = list(range(s_idx, e_idx))
        total_to_process = len(page_indices)

        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Parallel multi-core page processing
        num_workers = min(16, max(4, (os.cpu_count() or 4) * 2))
        doc_path_str = str(file_path.resolve())

        pages_results = [None] * total_to_process
        extracted_images = []
        completed_count = 0

        range_label = f"Pages {s_idx + 1} to {e_idx} of {total_doc_pages}" if (s_idx > 0 or e_idx < total_doc_pages) else f"all {total_doc_pages} pages"
        print(f"[*] Parsing {range_label} across {num_workers} parallel workers...", flush=True)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_slot = {
                executor.submit(self._process_single_page, doc_path_str, page_i, clean_stem, total_doc_pages): slot
                for slot, page_i in enumerate(page_indices)
            }
            for future in as_completed(future_to_slot):
                slot = future_to_slot[future]
                completed_count += 1
                try:
                    page_text, img_urls = future.result()
                    pages_results[slot] = page_text
                    if img_urls:
                        extracted_images.extend(img_urls)
                except Exception as e:
                    page_actual_num = page_indices[slot] + 1
                    pages_results[slot] = f"[Page {page_actual_num}]\n[Extraction error: {e}]"

                if progress_callback and (completed_count % 25 == 0 or completed_count == total_to_process):
                    progress_callback("parsing", completed_count, total_to_process, len(extracted_images))

                if completed_count % 100 == 0 or completed_count == total_to_process:
                    pct = (completed_count / total_to_process) * 100
                    print(f"  [PDF Progress] Processed {completed_count}/{total_to_process} pages ({pct:.1f}%) | Images extracted: {len(extracted_images)}", flush=True)

        pages_text = [p for p in pages_results if p is not None]
        full_text = "\n\n".join(pages_text)

        return ParsedDocument(
            filename=file_path.name,
            file_path=str(file_path.resolve()),
            file_type="pdf",
            text_content=full_text,
            pages=pages_text,
            metadata={
                "total_pages": total_to_process,
                "total_doc_pages": total_doc_pages,
                "page_range": f"{s_idx + 1}-{e_idx}",
                "char_count": len(full_text),
                "extracted_images": extracted_images,
                "diagram_count": len(extracted_images)
            }
        )
