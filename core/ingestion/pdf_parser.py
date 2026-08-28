import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
import pymupdf as fitz
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
    def detect_diagram_regions(cls, page: fitz.Page) -> List[Tuple[fitz.Rect, str, str]]:
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

    def parse(self, file_path: Path) -> ParsedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        clean_stem = "".join(c if c.isalnum() else "_" for c in file_path.stem)
        doc = fitz.open(str(file_path))
        total_pages = len(doc)

        pages_text = []
        extracted_diagrams = []

        for page_num in range(total_pages):
            page = doc[page_num]
            p_idx = page_num + 1
            page_sections = []

            # 1. Native Digital Text
            native_text = (page.get_text() or "").strip()

            # 2. Detect & Crop Exact Diagram Bounding Boxes
            diagram_regions = DiagramDetector.detect_diagram_regions(page)
            page_diagram_urls = []

            for d_idx, (diag_bbox, caption, diag_type) in enumerate(diagram_regions, start=1):
                clean_cap = "".join(c if c.isalnum() else "_" for c in caption[:25])
                diag_filename = f"{clean_stem}_p{p_idx}_diag{d_idx}_{clean_cap}.png"

                img_url = (
                    f"/api/sessions/{self.session_id}/images/{diag_filename}"
                    if self.session_id
                    else f"/images/{diag_filename}"
                )

                if self.output_images_dir:
                    target_path = self.output_images_dir / diag_filename
                    try:
                        # High-resolution 200 DPI crop of ONLY the diagram bounding box
                        pix = page.get_pixmap(dpi=200, clip=diag_bbox)
                        pix.save(str(target_path))
                        page_diagram_urls.append(img_url)
                        extracted_diagrams.append(img_url)

                        # Run RapidOCR on the cropped diagram
                        ocr = self._get_ocr()
                        ocr_res, _ = ocr(str(target_path))
                        diag_ocr_lines = [item[1].strip() for item in ocr_res if item[1].strip()] if ocr_res else []
                        diag_ocr_text = "\n".join(diag_ocr_lines)

                        diag_block = [
                            f"[DIAGRAM / FIGURE: {caption}]",
                            f"[Image URL: {img_url}]"
                        ]
                        if diag_ocr_text:
                            diag_block.append(f"[Diagram Content & OCR Text]:\n{diag_ocr_text}")

                        page_sections.append("\n".join(diag_block))

                    except Exception as e:
                        print(f"[*] Note cropping diagram bbox: {e}")

            # 3. If no specific diagram bounding box was isolated, render the page screenshot as fallback
            if not page_diagram_urls:
                full_page_filename = f"{clean_stem}_page_{p_idx}.png"
                full_page_url = (
                    f"/api/sessions/{self.session_id}/images/{full_page_filename}"
                    if self.session_id
                    else f"/images/{full_page_filename}"
                )
                if self.output_images_dir:
                    try:
                        pix = page.get_pixmap(dpi=150)
                        pix.save(str(self.output_images_dir / full_page_filename))
                        page_diagram_urls.append(full_page_url)
                    except Exception:
                        pass

                page_sections.append(f"[Page {p_idx} Overview Image]\n[Image URL: {full_page_url}]")

            if native_text:
                page_sections.append(native_text)

            full_page_content = "\n\n".join(page_sections)
            pages_text.append(full_page_content)

        doc.close()
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
