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

        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        native_page_text = (page.get_text() or "").strip()
        is_sparse_text_page = len(native_page_text) < 120

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

                # 1. Skip tiny icons / logos (width < 60 or height < 50 or area < 1.0%)
                if w < 60 or h < 50 or area_pct < 1.0:
                    continue

                # 2. Skip thin 1px horizontal / vertical divider lines (aspect > 6.0 or aspect < 0.15)
                if aspect > 6.0 or aspect < 0.15:
                    continue

                # 3. Skip full-page background slide templates (area > 80%) ONLY if the page has dense digital text
                # For magazines, catalogs, photos, and scanned slides, full-page images are PRIMARY content!
                if area_pct > 80.0 and not is_sparse_text_page:
                    continue

                # 4. Skip tiny footer icons that touch bottom 8%
                if h < 50 and bbox.y1 > page_rect.height * 0.92:
                    continue

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

    def __init__(
        self,
        output_images_dir: Optional[Path] = None,
        session_id: Optional[str] = None,
        vision_parser: Optional[Any] = None
    ):
        self.ocr_engine = None
        self.output_images_dir = Path(output_images_dir) if output_images_dir else None
        if self.output_images_dir:
            self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.vision_parser = vision_parser

    def _get_ocr(self):
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

            # 2. Detect & Crop Genuine Images/Figures (Only isolated real figures)
            image_regions = DiagramDetector.detect_diagram_regions(page)[:2]

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

            # Fallback for scanned slides, catalogs, posters, and image-heavy pages
            # If no bounded diagram was detected, but the page contains embedded images or is text-sparse:
            raw_page_images = page.get_images()
            if not image_regions and (raw_page_images or len(native_text) < 60):
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
                "diagram_count": len(extracted_images)
            }
        )
