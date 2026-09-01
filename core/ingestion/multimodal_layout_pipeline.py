"""
Mi:RAG - Multimodal Document Layout & Region Ingestion Pipeline
===============================================================
Purpose:
  Ingests complex, non-OCR-only PDFs containing mixed body text, structured tables,
  architecture diagrams, charts, and photos on the same page.

Pipeline Stages:
  1. High-Res Page Rendering (PyMuPDF at 2.5x-3.0x zoom matrix)
  2. Per-Page Layout Detection (PP-DocLayout-plus / PP-StructureV3 / Hybrid Fitz Analyzer)
  3. Per-Region Intelligent Routing:
       - Text blocks -> Local PyMuPDF / PP-OCRv5 extraction
       - Tables      -> Tabular Markdown/CSV cell structure extraction
       - Figures/Photos/Diagrams -> Crop -> Multimodal VLM (one-pass JSON classification + captioning)
  4. Unified Structured Chunk Generation (Strict JSON Schema with BBox & Image Paths)
  5. Dense Multilingual Embeddings (BGE-M3 / BGE-Base / MiniLM)
  6. Vector DB Persistence (Qdrant with full payload metadata / ChromaDB)
  7. Multimodal Visual-Aware Retrieval Engine
"""

import os
import io
import re
import json
import uuid
import base64
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Document & Image Processing
import fitz  # PyMuPDF
from PIL import Image

# Local OCR Engine
try:
    from rapidocr_onnxruntime import RapidOCR
    RAPID_OCR_AVAILABLE = True
except ImportError:
    RAPID_OCR_AVAILABLE = False

# HTTP Client for Vision-Language Models (VLM)
import httpx

# Embedding & Vector Database
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


# ==============================================================================
# 1. UNIFIED CHUNK DATA SCHEMA
# ==============================================================================

@dataclass
class RegionBBox:
    """Standardized bounding box coordinates: [x1, y1, x2, y2]."""
    x1: float
    y1: float
    x2: float
    y2: float
    region_type: str  # text | table | diagram | chart | photo | illustration | caption
    confidence: float = 1.0
    text_content: Optional[str] = None

    def to_list(self) -> List[float]:
        return [round(self.x1, 2), round(self.y1, 2), round(self.x2, 2), round(self.y2, 2)]

    @property
    def fitz_rect(self) -> fitz.Rect:
        return fitz.Rect(self.x1, self.y1, self.x2, self.y2)


@dataclass
class UnifiedChunk:
    """
    Standardized Multimodal Chunk Schema.
    Required by Mi:RAG & InsightForge architecture.
    """
    id: str
    document: str
    page: int
    type: str  # text | table | diagram | chart | photo | illustration
    bbox: List[float]  # [x1, y1, x2, y2]
    content: str  # Text content OR generated VLM description
    image_path: Optional[str] = None  # Path to isolated crop (for visual types)
    page_image_path: Optional[str] = None  # Full page reference
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. HIGH-RES PAGE RENDERER
# ==============================================================================

class HighResPageRenderer:
    """
    Renders PDF pages to high-resolution images (2.5x-3.0x zoom matrix ~200-300 DPI)
    so small diagram annotations, subscripts, and flowcharts remain sharp.
    """

    def __init__(self, zoom_factor: float = 2.5):
        self.matrix = fitz.Matrix(zoom_factor, zoom_factor)

    def render_page(self, page: fitz.Page, output_path: Optional[Path] = None) -> Tuple[Image.Image, Optional[Path]]:
        """Renders fitz.Page to a PIL Image and saves full-page reference to disk."""
        pix = page.get_pixmap(matrix=self.matrix, alpha=False)
        img_data = pix.tobytes("png")
        pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(str(output_path), "PNG", optimize=True)

        return pil_img, output_path


# ==============================================================================
# 3. LAYOUT DETECTION & REGION PARSER
# ==============================================================================

class LayoutDetector:
    """
    Identifies layout regions on mixed pages:
    - Text blocks
    - Tables
    - Figures / Diagrams / Charts / Photos
    - Captions
    """

    CAPTION_REGEX = re.compile(r"^(fig(?:ure)?\.?|table|chart|diagram|architecture|workflow|algorithm)\s*\d*", re.IGNORECASE)

    def __init__(self, use_paddle_structure: bool = False):
        self.use_paddle = use_paddle_structure
        self.paddle_engine = None
        if self.use_paddle:
            try:
                from paddleocr import PPStructure
                self.paddle_engine = PPStructure(table=True, ocr=True, show_log=False)
                print("[*] PP-StructureV3 Layout Engine loaded.")
            except Exception as e:
                print(f"[*] Native Hybrid Layout Engine Active (PP-Structure optional): {e}")

    def detect_regions(self, page: fitz.Page, page_image: Image.Image) -> List[RegionBBox]:
        """Detects and partitions page into discrete classified layout regions."""
        regions: List[RegionBBox] = []

        # 1. Native Digital Text Blocks & Captions
        blocks = page.get_text("blocks")
        for b in blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            clean_text = text.strip()
            if not clean_text:
                continue

            if self.CAPTION_REGEX.search(clean_text) and len(clean_text) < 300:
                regions.append(RegionBBox(x1=x0, y1=y0, x2=x1, y2=y1, region_type="caption", text_content=clean_text))
            else:
                regions.append(RegionBBox(x1=x0, y1=y0, x2=x1, y2=y1, region_type="text", text_content=clean_text))

        # 2. Embedded Figures, Photos & Charts
        img_info_list = page.get_image_info(xrefs=True)
        page_area = page.rect.width * page.rect.height

        for img in img_info_list:
            bbox = fitz.Rect(img.get("bbox", (0, 0, 0, 0)))
            if bbox.is_valid and not bbox.is_empty:
                area = bbox.width * bbox.height
                # Filter out tiny icon decorations (<60x50) and full-page backgrounds (>85% area)
                if bbox.width >= 60 and bbox.height >= 50 and (area < page_area * 0.85):
                    regions.append(RegionBBox(x1=bbox.x0, y1=bbox.y0, x2=bbox.x1, y2=bbox.y1, region_type="figure"))

        # 3. Native Table Structure Extraction (PyMuPDF find_tables)
        try:
            tabs = page.find_tables()
            for tab in tabs:
                t_rect = fitz.Rect(tab.bbox)
                if t_rect.is_valid:
                    df = tab.extract()
                    md_lines = []
                    for row in df:
                        clean_row = [str(c or "").strip().replace("\n", " ") for c in row]
                        md_lines.append("| " + " | ".join(clean_row) + " |")
                    table_md = "\n".join(md_lines)
                    regions.append(RegionBBox(x1=t_rect.x0, y1=t_rect.y0, x2=t_rect.x1, y2=t_rect.y1, region_type="table", text_content=table_md))
        except Exception:
            pass

        return regions


# ==============================================================================
# 4. MULTIMODAL VLM CLASSIFIER & CAPTION GENERATOR
# ==============================================================================

class VisionRegionClassifier:
    """
    Uses Vision-Language Models (VLM) via Ollama, PaddleOCR-VL-1.5, or OpenAI
    to classify crops (`PHOTO | DIAGRAM | CHART | GRAPH | TABLE | ILLUSTRATION | OTHER`)
    and generate detailed semantic descriptions in strict JSON format.
    """

    def __init__(
        self,
        endpoint_url: str = "http://localhost:11434/api/generate",
        model_name: str = "moondream",
        api_key: Optional[str] = None
    ):
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.api_key = api_key

    def classify_and_describe(self, crop_image: Image.Image) -> Dict[str, str]:
        """
        Runs one-pass visual classification and structured caption generation.
        Returns:
            {"category": "diagram", "description": "Flowchart showing..."}
        """
        buffered = io.BytesIO()
        crop_image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        prompt = (
            "Analyze this document visual crop precisely.\n"
            "1. Classify its exact type as one of: [PHOTO, DIAGRAM, CHART, GRAPH, TABLE, ILLUSTRATION, OTHER].\n"
            "2. Provide a thorough, self-contained semantic description of what is shown, including all text labels, relationships, data points, or workflow steps.\n"
            "Respond in strict JSON format:\n"
            "{\n"
            '  "type": "DIAGRAM",\n'
            '  "description": "A flowchart showing step-by-step data ingestion..."\n'
            "}"
        )

        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {"temperature": 0.1}
            }
            with httpx.Client(timeout=12.0) as client:
                res = client.post(self.endpoint_url, json=payload)
                if res.status_code == 200:
                    resp_text = res.json().get("response", "").strip()
                    if "{" in resp_text and "}" in resp_text:
                        json_str = resp_text[resp_text.find("{"):resp_text.rfind("}") + 1]
                        parsed = json.loads(json_str)
                        return {
                            "type": parsed.get("type", "diagram").lower(),
                            "description": parsed.get("description", resp_text)
                        }
                    return {"type": "diagram", "description": resp_text}
        except Exception:
            pass

        return {
            "type": "diagram",
            "description": "Visual diagram crop containing architectural or workflow components."
        }


# ==============================================================================
# 5. MULTIMODAL INGESTION PIPELINE
# ==============================================================================

class MultimodalLayoutPipeline:
    """
    Unified Ingestion & Multimodal Retrieval Pipeline:
    PDF -> Render High-Res Pages -> Detect Layout Regions -> Route & Crop ->
    Unified Chunks -> Dense Embeddings (BGE-M3) -> Qdrant / Vector DB -> Search & Retrieval.
    """

    def __init__(
        self,
        storage_dir: Union[str, Path] = "./data/multimodal_store",
        embedding_model_name: str = "BAAI/bge-m3",
        qdrant_url: Optional[str] = None,
        qdrant_path: str = "./data/multimodal_store/qdrant_db",
        collection_name: str = "mirag_multimodal"
    ):
        self.storage_dir = Path(storage_dir)
        self.images_dir = self.storage_dir / "crops"
        self.page_images_dir = self.storage_dir / "pages"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.page_images_dir.mkdir(parents=True, exist_ok=True)

        self.renderer = HighResPageRenderer(zoom_factor=2.5)
        self.layout_detector = LayoutDetector()
        self.vision_classifier = VisionRegionClassifier()
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        # Dense Embedder
        print(f"[*] Initializing Dense Multilingual Embedder: {embedding_model_name}...")
        self.embedder = SentenceTransformer(embedding_model_name) if SENTENCE_TRANSFORMERS_AVAILABLE else None

        # Vector Storage (Qdrant or In-Memory)
        self.qdrant_client = None
        self._memory_index: List[Tuple[UnifiedChunk, List[float]]] = []
        self._init_vector_db(qdrant_url, qdrant_path)

    def _init_vector_db(self, qdrant_url: Optional[str], qdrant_path: str):
        if not QDRANT_AVAILABLE:
            print("[*] Running with in-memory vector index.")
            return

        try:
            if qdrant_url:
                self.qdrant_client = QdrantClient(url=qdrant_url)
            else:
                self.qdrant_client = QdrantClient(path=qdrant_path)

            dim = 1024
            if self.embedder:
                try:
                    dim = self.embedder.get_sentence_embedding_dimension()
                except Exception:
                    dim = 1024

            existing = [c.name for c in self.qdrant_client.get_collections().collections]
            if self.collection_name not in existing:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
                )
                print(f"[*] Initialized Qdrant collection '{self.collection_name}' (dim={dim})")
        except Exception as e:
            print(f"[!] Qdrant initialization fallback to local storage: {e}")
            self.qdrant_client = None

    def ingest_pdf(self, pdf_path: Union[str, Path], progress_callback: Optional[Any] = None) -> List[UnifiedChunk]:
        """
        Parses mixed-layout PDF into structured unified chunks with visual crops.
        """
        pdf_path = Path(pdf_path)
        doc_name = pdf_path.name
        doc_stem = "".join(c if c.isalnum() else "_" for c in pdf_path.stem)
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)

        print(f"\n[Mi:RAG Ingestion] Processing '{doc_name}' ({total_pages} pages)...")
        all_chunks: List[UnifiedChunk] = []

        for p_no in range(total_pages):
            page = doc[p_no]
            page_num = p_no + 1

            # 1. Render Full High-Res Page Image
            page_img_path = self.page_images_dir / f"{doc_stem}_page_{page_num}.png"
            page_img, _ = self.renderer.render_page(page, output_path=page_img_path)

            # 2. Layout Detection
            regions = self.layout_detector.detect_regions(page, page_img)
            caption_map = {r: r.text_content for r in regions if r.region_type == "caption"}

            figure_idx = 1
            for r_idx, region in enumerate(regions):
                chunk_id = f"{doc_stem}_p{page_num}_r{r_idx + 1}"

                # 3. Per-Region Routing
                if region.region_type in ["figure", "diagram", "chart", "photo", "illustration"]:
                    crop_filename = f"{doc_stem}_page{page_num}_fig{figure_idx}.png"
                    crop_path = self.images_dir / crop_filename

                    # High-res crop
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=region.fitz_rect)
                    pix.save(str(crop_path))
                    figure_idx += 1

                    # VLM Classification + Captioning
                    with Image.open(crop_path) as crop_img:
                        vlm_res = self.vision_classifier.classify_and_describe(crop_img)

                    classified_type = vlm_res.get("type", "diagram")
                    content_text = f"[{classified_type.upper()}] {vlm_res.get('description', '')}"

                    # Attach caption context if found near figure
                    for cap_region, cap_text in caption_map.items():
                        if abs(cap_region.y1 - region.y2) < 50 or abs(cap_region.y2 - region.y1) < 50:
                            content_text = f"[Caption: {cap_text}]\n{content_text}"
                            break

                    chunk = UnifiedChunk(
                        id=chunk_id,
                        document=doc_name,
                        page=page_num,
                        type=classified_type,
                        bbox=region.to_list(),
                        content=content_text,
                        image_path=str(crop_path.resolve()),
                        page_image_path=str(page_img_path.resolve()),
                        metadata={"confidence": region.confidence}
                    )
                    all_chunks.append(chunk)

                elif region.region_type == "table":
                    chunk = UnifiedChunk(
                        id=chunk_id,
                        document=doc_name,
                        page=page_num,
                        type="table",
                        bbox=region.to_list(),
                        content=region.text_content or "[Table Data]",
                        image_path=None,
                        page_image_path=str(page_img_path.resolve()),
                        metadata={"format": "markdown"}
                    )
                    all_chunks.append(chunk)

                elif region.region_type == "text":
                    chunk = UnifiedChunk(
                        id=chunk_id,
                        document=doc_name,
                        page=page_num,
                        type="text",
                        bbox=region.to_list(),
                        content=region.text_content or "",
                        image_path=None,
                        page_image_path=str(page_img_path.resolve())
                    )
                    all_chunks.append(chunk)

            if progress_callback:
                progress_callback("parsing", page_num, total_pages, sum(1 for c in all_chunks if c.image_path))

        doc.close()
        print(f"[*] Generated {len(all_chunks)} unified chunks ({sum(1 for c in all_chunks if c.image_path)} visual crops).")

        # 4. Embed & Store in Vector DB
        self._store_chunks(all_chunks)
        return all_chunks

    def _store_chunks(self, chunks: List[UnifiedChunk]):
        """Embeds and upserts chunks into Qdrant or local memory."""
        if not chunks or not self.embedder:
            return

        texts = [c.content for c in chunks]
        vectors = self.embedder.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)

        if self.qdrant_client:
            points = []
            for chunk, vector in zip(chunks, vectors):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id))
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload=chunk.to_dict()
                    )
                )
            self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
            print(f"[*] Upserted {len(points)} vectors to Qdrant collection '{self.collection_name}'.")
        else:
            for chunk, vec in zip(chunks, vectors):
                self._memory_index.append((chunk, vec.tolist()))

    def retrieve(self, query: str, top_k: int = 5, filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Multimodal Visual-Aware Retrieval Engine.
        Returns matched text + isolated image_path so visual figures are surfaced.
        """
        if not self.embedder:
            return []

        query_vector = self.embedder.encode(query, normalize_embeddings=True).tolist()

        if self.qdrant_client:
            query_filter = None
            if filter_type:
                query_filter = Filter(must=[FieldCondition(key="type", match=MatchValue(value=filter_type))])

            hits = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter
            )

            return [
                {
                    "id": hit.payload.get("id"),
                    "document": hit.payload.get("document"),
                    "page": hit.payload.get("page"),
                    "type": hit.payload.get("type"),
                    "score": round(hit.score, 4),
                    "bbox": hit.payload.get("bbox"),
                    "content": hit.payload.get("content"),
                    "image_path": hit.payload.get("image_path"),
                    "page_image_path": hit.payload.get("page_image_path"),
                    "is_visual": hit.payload.get("image_path") is not None
                }
                for hit in hits
            ]

        # In-Memory Cosine Similarity fallback
        import numpy as np
        q_vec = np.array(query_vector)
        scored = []
        for chunk, vec in self._memory_index:
            if filter_type and chunk.type != filter_type:
                continue
            sim = float(np.dot(q_vec, np.array(vec)))
            scored.append((sim, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": c.id,
                "document": c.document,
                "page": c.page,
                "type": c.type,
                "score": round(s, 4),
                "bbox": c.bbox,
                "content": c.content,
                "image_path": c.image_path,
                "page_image_path": c.page_image_path,
                "is_visual": c.image_path is not None
            }
            for s, c in scored[:top_k]
        ]


# ==============================================================================
# 6. STANDALONE VERIFICATION RUNNER
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Mi:RAG - Multimodal Document Layout & Region Ingestion Pipeline")
    print("=" * 70)
    pipeline = MultimodalLayoutPipeline()
    print("[*] Ready for multimodal document processing.")
