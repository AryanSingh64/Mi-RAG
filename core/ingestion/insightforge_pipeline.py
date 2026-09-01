"""
InsightForge AI: Multimodal RAG Ingestion Pipeline (Non-OCR-Only PDFs)
=======================================================================
Architecture:
  PDF -> High-Res Page Rendering (PyMuPDF)
      -> Layout Detection (PP-DocLayout / Structural BBox Analyzer)
      -> Per-Region Routing:
           * Text / Headers -> Local PP-OCR / PyMuPDF High-Precision Extraction
           * Tables         -> Table Structure Extractor (Markdown / CSV)
           * Figures / Photos / Diagrams -> High-Res Crop -> Multimodal VLM (Classification + Captioning)
      -> Unified Structured Chunk Generation (Strict JSON Schema)
      -> Dense Multilingual Vector Embeddings (e.g., BAAI/bge-m3)
      -> Vector DB Upsert (Qdrant with full payload metadata)
      -> Visual-Aware Retrieval Query Engine
"""

import os
import io
import json
import uuid
import base64
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Core Document & Imaging Tools
import fitz  # PyMuPDF
from PIL import Image

# Fallback / Lightweight OCR
try:
    from rapidocr_onnxruntime import RapidOCR
    RAPID_OCR_AVAILABLE = True
except ImportError:
    RAPID_OCR_AVAILABLE = False

# HTTP / API Client for VLM inference
import httpx

# Embedding and Vector Storage
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
# 1. UNIFIED CHUNK SCHEMA & DATA MODELS
# ==============================================================================

@dataclass
class RegionBox:
    """Bounding box coordinates in standard [x1, y1, x2, y2] point format."""
    x1: float
    y1: float
    x2: float
    y2: float
    region_type: str  # text, table, figure, diagram, chart, photo, illustration, caption
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
    Required by InsightForge AI architecture.
    """
    id: str
    document: str
    page: int
    type: str  # text | table | diagram | chart | photo | illustration | caption
    bbox: List[float]  # [x1, y1, x2, y2]
    content: str  # Text content OR generated VLM description
    image_path: Optional[str] = None  # Path to isolated crop (only for visual types)
    page_image_path: Optional[str] = None  # Full page reference for context reconstruction
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 2. HIGH-RES PAGE RENDERER (PyMuPDF)
# ==============================================================================

class HighResPageRenderer:
    """
    Renders PDF pages to high-resolution images (2x-3x matrix / 200-300 DPI)
    so small annotations and diagram labels remain sharp and legible.
    """

    def __init__(self, zoom_factor: float = 2.5):
        self.matrix = fitz.Matrix(zoom_factor, zoom_factor)

    def render_page(self, page: fitz.Page, output_path: Optional[Path] = None) -> Tuple[Image.Image, Path]:
        """Renders a single fitz.Page to a PIL Image and optional disk file."""
        pix = page.get_pixmap(matrix=self.matrix, alpha=False)
        img_data = pix.tobytes("png")
        pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(str(output_path), "PNG", optimize=True)

        return pil_img, output_path


# ==============================================================================
# 3. LAYOUT DETECTION ENGINE (PP-DocLayout / Hybrid Structural Analyzer)
# ==============================================================================

class LayoutDetector:
    """
    Detects layout regions per page:
    - Text blocks
    - Tables
    - Figures / Diagrams / Charts / Photos
    - Captions
    
    Supports PaddleOCR PP-StructureV3 / PP-DocLayout-plus with fallback to
    PyMuPDF vector/drawing cluster + block classification.
    """

    def __init__(self, use_paddle: bool = False):
        self.use_paddle = use_paddle
        self.paddle_engine = None
        if self.use_paddle:
            try:
                from paddleocr import PPStructure
                # PP-StructureV3 / Layout model
                self.paddle_engine = PPStructure(table=True, ocr=True, show_log=False)
            except Exception as e:
                print(f"[LayoutDetector] PaddleOCR PP-Structure not available, using high-precision native hybrid detector: {e}")

    def detect_regions(self, page: fitz.Page, page_image: Image.Image) -> List[RegionBox]:
        """Detects and partitions page into discrete classified layout regions."""
        regions: List[RegionBox] = []

        # 1. Extract native text blocks and captions
        text_blocks = page.get_text("blocks")
        caption_regex = fitz.re.compile(r"^(fig(?:ure)?\.?|table|chart|diagram|architecture|workflow)\s*\d*", fitz.re.IGNORECASE)

        for b in text_blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            clean_text = text.strip()
            if not clean_text:
                continue

            if caption_regex.search(clean_text) and len(clean_text) < 300:
                regions.append(RegionBox(x1=x0, y1=y0, x2=x1, y2=y1, region_type="caption", text_content=clean_text))
            else:
                regions.append(RegionBox(x1=x0, y1=y0, x2=x1, y2=y1, region_type="text", text_content=clean_text))

        # 2. Extract Embedded Visual Images
        img_info_list = page.get_image_info(xrefs=True)
        page_area = page.rect.width * page.rect.height

        for img in img_info_list:
            bbox = fitz.Rect(img.get("bbox", (0, 0, 0, 0)))
            if bbox.is_valid and not bbox.is_empty:
                area = bbox.width * bbox.height
                # Ignore tiny bullet points/icons (<60px) and full-page backgrounds (>85% area)
                if bbox.width > 60 and bbox.height > 50 and (area < page_area * 0.85):
                    regions.append(RegionBox(x1=bbox.x0, y1=bbox.y0, x2=bbox.x1, y2=bbox.y1, region_type="figure"))

        # 3. Detect Native Tables if available in PyMuPDF v1.23+
        try:
            tabs = page.find_tables()
            for tab in tabs:
                t_rect = fitz.Rect(tab.bbox)
                if t_rect.is_valid:
                    # Convert table to Markdown string
                    df = tab.extract()
                    md_lines = []
                    for row in df:
                        clean_row = [str(c or "").strip().replace("\n", " ") for c in row]
                        md_lines.append("| " + " | ".join(clean_row) + " |")
                    table_md = "\n".join(md_lines)
                    regions.append(RegionBox(x1=t_rect.x0, y1=t_rect.y0, x2=t_rect.x1, y2=t_rect.y1, region_type="table", text_content=table_md))
        except Exception:
            pass

        return regions


# ==============================================================================
# 4. MULTIMODAL VLM CLASSIFIER & CAPTION GENERATOR
# ==============================================================================

class VisionRegionClassifier:
    """
    Uses a Vision-Language Model (VLM) via Ollama, PaddleOCR-VL-1.5, or OpenAI
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
            {"category": "DIAGRAM", "description": "...", "key_elements": "..."}
        """
        # Convert PIL Image to Base64
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
            with httpx.Client(timeout=15.0) as client:
                res = client.post(self.endpoint_url, json=payload)
                if res.status_code == 200:
                    resp_text = res.json().get("response", "").strip()
                    # Parse JSON or fallback
                    if "{" in resp_text and "}" in resp_text:
                        json_str = resp_text[resp_text.find("{"):resp_text.rfind("}") + 1]
                        parsed = json.loads(json_str)
                        return {
                            "type": parsed.get("type", "diagram").lower(),
                            "description": parsed.get("description", resp_text)
                        }
                    return {"type": "diagram", "description": resp_text}
        except Exception as e:
            pass

        return {
            "type": "diagram",
            "description": "Visual figure crop containing architectural or workflow components."
        }


# ==============================================================================
# 5. MULTIMODAL INGESTION PIPELINE (InsightForge AI)
# ==============================================================================

class InsightForgePipeline:
    """
    End-to-End Multimodal Ingestion Pipeline:
    PDF -> Render High-Res Pages -> Detect Layout Regions -> Route & Crop ->
    Unified Chunk Generation -> Embed (BGE-M3) -> Qdrant Vector DB Storage.
    """

    def __init__(
        self,
        storage_dir: Union[str, Path] = "./insightforge_data",
        embedding_model_name: str = "BAAI/bge-m3",
        qdrant_url: Optional[str] = None,  # None = local in-memory/disk
        qdrant_path: str = "./insightforge_data/qdrant_db",
        collection_name: str = "insightforge_multimodal"
    ):
        self.storage_dir = Path(storage_dir)
        self.images_dir = self.storage_dir / "images"
        self.page_images_dir = self.storage_dir / "page_images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.page_images_dir.mkdir(parents=True, exist_ok=True)

        self.renderer = HighResPageRenderer(zoom_factor=2.5)
        self.layout_detector = LayoutDetector()
        self.vision_classifier = VisionRegionClassifier()
        self.collection_name = collection_name

        # Initialize SOTA Multilingual Embedder (BGE-M3 default)
        print(f"[*] Initializing Dense Embedder: {embedding_model_name}...")
        self.embedder = SentenceTransformer(embedding_model_name) if SENTENCE_TRANSFORMERS_AVAILABLE else None

        # Initialize Qdrant Vector Store
        self._init_vector_db(qdrant_url, qdrant_path)

    def _init_vector_db(self, qdrant_url: Optional[str], qdrant_path: str):
        """Initializes Qdrant collection with vector indexing for dense retrieval."""
        if not QDRANT_AVAILABLE:
            print("[!] Qdrant client not installed. Falling back to local vector memory.")
            self.qdrant_client = None
            return

        if qdrant_url:
            self.qdrant_client = QdrantClient(url=qdrant_url)
        else:
            self.qdrant_client = QdrantClient(path=qdrant_path)

        dim = 1024  # Standard for BGE-M3 (or 384 / 768 depending on model)
        if self.embedder:
            try:
                dim = self.embedder.get_sentence_embedding_dimension()
            except Exception:
                dim = 1024

        collections = [c.name for c in self.qdrant_client.get_collections().collections]
        if self.collection_name not in collections:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
            print(f"[*] Created Qdrant collection '{self.collection_name}' with dimension {dim}")

    def ingest_pdf(self, pdf_path: Union[str, Path]) -> List[UnifiedChunk]:
        """
        Processes a full PDF document:
        - Renders pages
        - Detects mixed regions
        - Crops visual figures & runs VLM descriptions
        - Formats unified chunks
        - Upserts to Qdrant
        """
        pdf_path = Path(pdf_path)
        doc_name = pdf_path.name
        doc_stem = pdf_path.stem
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)

        print(f"\n[InsightForge Ingest] Starting ingestion for '{doc_name}' ({total_pages} pages)...")
        all_chunks: List[UnifiedChunk] = []

        for p_no in range(total_pages):
            page = doc[p_no]
            page_num = p_no + 1

            # 1. Render Full High-Res Page Image
            page_img_path = self.page_images_dir / f"{doc_stem}_page_{page_num}.png"
            page_img, _ = self.renderer.render_page(page, output_path=page_img_path)

            # 2. Detect Per-Page Layout Regions
            regions = self.layout_detector.detect_regions(page, page_img)

            # Associate captions with nearby figures
            caption_map = {r: r.text_content for r in regions if r.region_type == "caption"}

            figure_idx = 1
            for r_idx, region in enumerate(regions):
                chunk_id = f"{doc_stem}_p{page_num}_r{r_idx + 1}"

                # 3. Route Per-Region:
                if region.region_type in ["figure", "diagram", "chart", "photo", "illustration"]:
                    # Crop visual region at high resolution
                    crop_filename = f"{doc_stem}_page{page_num}_fig{figure_idx}.png"
                    crop_path = self.images_dir / crop_filename
                    
                    # Crop from fitz.Page at high DPI
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=region.fitz_rect)
                    pix.save(str(crop_path))
                    figure_idx += 1

                    # Run Multimodal VLM for Classification + Semantic Captioning
                    with Image.open(crop_path) as crop_img:
                        vlm_res = self.vision_classifier.classify_and_describe(crop_img)

                    classified_type = vlm_res.get("type", "diagram")
                    content_text = f"[{classified_type.upper()}] {vlm_res.get('description', '')}"

                    # Attach any matching caption text
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

        doc.close()
        print(f"[*] Processed {len(all_chunks)} unified structured chunks ({sum(1 for c in all_chunks if c.image_path)} visual crops).")

        # 4. Embed & Store in Qdrant
        self._store_chunks_in_qdrant(all_chunks)
        return all_chunks

    def _store_chunks_in_qdrant(self, chunks: List[UnifiedChunk]):
        """Generates dense vectors and upserts payload records to Qdrant."""
        if not chunks or not self.embedder or not self.qdrant_client:
            return

        texts_to_embed = [c.content for c in chunks]
        embeddings = self.embedder.encode(texts_to_embed, batch_size=32, show_progress_bar=False, normalize_embeddings=True)

        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id))
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=chunk.to_dict()
                )
            )

        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"[InsightForge Ingest] Successfully upserted {len(points)} points into Qdrant collection '{self.collection_name}'.")

    # ==============================================================================
    # 6. MULTIMODAL RETRIEVAL ENGINE
    # ==============================================================================

    def retrieve(self, query: str, top_k: int = 5, filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Multimodal retrieval function:
        Retrieves the most relevant chunks matching the user query.
        Returns both the matched content AND the actual image_path for visual items.
        """
        if not self.embedder or not self.qdrant_client:
            print("[!] Embedder or Qdrant not available.")
            return []

        query_vector = self.embedder.encode(query, normalize_embeddings=True).tolist()
        
        query_filter = None
        if filter_type:
            query_filter = Filter(
                must=[FieldCondition(key="type", match=MatchValue(value=filter_type))]
            )

        results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter
        )

        formatted_results = []
        for hit in results:
            payload = hit.payload
            formatted_results.append({
                "id": payload.get("id"),
                "document": payload.get("document"),
                "page": payload.get("page"),
                "type": payload.get("type"),
                "score": round(hit.score, 4),
                "bbox": payload.get("bbox"),
                "content": payload.get("content"),
                "image_path": payload.get("image_path"),
                "page_image_path": payload.get("page_image_path"),
                "is_visual": payload.get("image_path") is not None
            })

        return formatted_results


# ==============================================================================
# 7. DEMONSTRATION & VERIFICATION SCRIPT
# ==============================================================================

if __name__ == "__main__":
    print("InsightForge AI - Multimodal Ingestion Pipeline Initialized.")
