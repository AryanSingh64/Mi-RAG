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
        Analyzes page to return all genuine diagrams and figures: (cropped_bbox, caption_text, diagram_type).
        Extracts both embedded raster images AND vector diagrams (charts, plots, flowcharts).
        """
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        native_page_text = (page.get_text() or "").strip()
        is_sparse_text_page = len(native_page_text) < 120

        # 1. Find Figure/Diagram captions in text blocks
        blocks = page.get_text("blocks")
        caption_blocks = []
        for b in blocks:
            text = b[4].strip()
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            if lines and cls.CAPTION_REGEX.search(lines[0]) and len(text) < 450:
                b_rect = fitz.Rect(b[:4])
                caption_blocks.append((b_rect, lines[0]))

        diagrams = []
        covered_rects = []

        # 2. Collect genuine standalone raster image bounding boxes
        img_info_list = page.get_image_info(xrefs=True)
        img_rects = []
        for img_info in img_info_list:
            bbox = fitz.Rect(img_info.get("bbox", (0, 0, 0, 0)))
            if bbox.is_valid and not bbox.is_empty:
                w, h = bbox.width, bbox.height
                area = w * h
                aspect = w / max(1.0, h)
                # Filter out tiny icon decorations / bullets (<22x22 or <450 area)
                if w < 22 or h < 22 or (area < 450):
                    continue
                # Filter out extreme thin divider rules
                if aspect > 25.0 or aspect < 0.04:
                    continue
                # Filter out background slide templates (>88% area on text-dense pages)
                if (area / page_area) > 0.88 and not is_sparse_text_page:
                    continue
                # Deduplicate identical boxes
                if any(abs(bbox.x0 - cr.x0) < 6 and abs(bbox.y0 - cr.y0) < 6 for cr in img_rects):
                    continue
                img_rects.append(bbox)

        # Also inspect page.get_images() for xrefs that have on-page rects not in image_info
        raw_images = page.get_images(full=True)
        for img_tuple in raw_images:
            xref = img_tuple[0]
            try:
                for r in page.get_image_rects(xref):
                    if r.is_valid and not r.is_empty:
                        if r.width >= 22 and r.height >= 22 and (r.width * r.height >= 450):
                            if not any(abs(r.x0 - cr.x0) < 6 and abs(r.y0 - cr.y0) < 6 for cr in img_rects):
                                img_rects.append(r)
            except Exception:
                pass

        # 3. Associate raster images with nearest captions
        for idx, img_bbox in enumerate(img_rects, start=1):
            matched_caption = ""
            best_dist = 180.0
            for c_rect, c_text in caption_blocks:
                dist = min(abs(c_rect.y0 - img_bbox.y1), abs(img_bbox.y0 - c_rect.y1))
                if dist < best_dist:
                    best_dist = dist
                    matched_caption = c_text

            padded_rect = fitz.Rect(
                max(0, img_bbox.x0 - 6),
                max(0, img_bbox.y0 - 6),
                min(page_rect.width, img_bbox.x1 + 6),
                min(page_rect.height, img_bbox.y1 + 6)
            )
            diag_name = matched_caption.split("\n")[0][:70] if matched_caption else f"Figure {idx}"
            diagrams.append((padded_rect, diag_name, "raster_figure"))
            covered_rects.append(img_bbox)

        # 4. Extract Vector Diagrams, Charts & Plots (drawings) for remaining unassigned captions
        drawings = page.get_drawings()
        if drawings and caption_blocks:
            for c_rect, c_text in caption_blocks:
                # Check if this caption was already matched to a raster image
                already_covered = any(abs(c_rect.y0 - cr.y1) < 40 or abs(cr.y0 - c_rect.y1) < 40 for cr in covered_rects)
                if already_covered:
                    continue

                # Check vector drawings located above caption (standard for figures)
                d_above = [d for d in drawings if d['rect'].y1 <= c_rect.y0 + 6 and d['rect'].y0 >= c_rect.y0 - 450]
                if d_above:
                    vbox = fitz.Rect(d_above[0]['rect'])
                    for d in d_above[1:]:
                        vbox |= d['rect']
                    if vbox.width >= 50 and vbox.height >= 35:
                        padded_vbox = fitz.Rect(
                            max(0, vbox.x0 - 6),
                            max(0, vbox.y0 - 6),
                            min(page_rect.width, vbox.x1 + 6),
                            min(page_rect.height, vbox.y1 + 6)
                        )
                        cap_name = c_text.split("\n")[0][:70]
                        diagrams.append((padded_vbox, cap_name, "vector_figure"))
                        covered_rects.append(padded_vbox)
                        continue

                # Check vector drawings located below caption (standard for tables/top titles)
                d_below = [d for d in drawings if d['rect'].y0 >= c_rect.y1 - 6 and d['rect'].y1 <= c_rect.y1 + 450]
                if d_below:
                    vbox = fitz.Rect(d_below[0]['rect'])
                    for d in d_below[1:]:
                        vbox |= d['rect']
                    if vbox.width >= 50 and vbox.height >= 35:
                        padded_vbox = fitz.Rect(
                            max(0, vbox.x0 - 6),
                            max(0, vbox.y0 - 6),
                            min(page_rect.width, vbox.x1 + 6),
                            min(page_rect.height, vbox.y1 + 6)
                        )
                        cap_name = c_text.split("\n")[0][:70]
                        diagrams.append((padded_vbox, cap_name, "vector_figure"))
                        covered_rects.append(padded_vbox)

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

            # 3. Direct Image XObject Extraction for any embedded image not cropped via on-page bboxes
            try:
                for img_tuple in page.get_images(full=True):
                    xref = img_tuple[0]
                    img_dict = thread_doc.extract_image(xref)
                    if img_dict:
                        w, h = img_dict.get("width", 0), img_dict.get("height", 0)
                        if w >= 50 and h >= 50 and (w * h >= 4000):
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
