import base64
import io
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
import httpx
from core.ingestion.base import BaseDocumentParser, ParsedDocument


class VisionImageParser(BaseDocumentParser):
    """
    Multimodal Image Parser.
    Extracts text using RapidOCR (with alpha channel flattening) AND generates a rich semantic description via Ollama Vision.
    """

    def __init__(self, ollama_url: str = "http://localhost:11434", vision_model: str = "moondream"):
        self.ocr_engine = RapidOCR()
        self.ollama_url = ollama_url
        self.vision_model = vision_model

    def _prepare_image(self, image_path: Path) -> tuple[np.ndarray, str]:
        """Loads image, handles transparency with clean white background, and returns RGB numpy array and base64 string."""
        with Image.open(image_path) as img:
            # Handle alpha transparency by converting to white background
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgba = img.convert("RGBA")
                bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                bg.paste(rgba, (0, 0), rgba)
                rgb_img = bg.convert("RGB")
            else:
                rgb_img = img.convert("RGB")

            # Convert to numpy array for RapidOCR
            np_img = np.array(rgb_img)

            # Convert to base64 for Ollama Vision
            buffered = io.BytesIO()
            rgb_img.save(buffered, format="JPEG", quality=95)
            b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return np_img, b64_str

    def _extract_via_ocr(self, np_img: np.ndarray) -> str:
        """Fast local CPU OCR extraction on normalized RGB numpy image."""
        try:
            result, _ = self.ocr_engine(np_img)
            if not result:
                return ""
            lines = [item[1].strip() for item in result if item[1].strip()]
            return "\n".join(lines)
        except Exception:
            return ""

    def _extract_via_ollama_vision(self, b64_image: str) -> Optional[str]:
        """Generates detailed semantic description of image contents using Ollama Vision."""
        try:
            prompt = (
                "Deep visual inspection: "
                "1. Transcribe ALL visible text, words, logos, abbreviations, numbers, and labels exactly as written. "
                "2. Describe the visual layout, typography, font style (e.g. cursive, block, 3D, chrome), color palette, and background. "
                "3. Explain what this image, logo, chart, or document represents."
            )

            payload = {
                "model": self.vision_model,
                "prompt": prompt,
                "images": [b64_image],
                "stream": False
            }

            with httpx.Client(timeout=60.0) as client:
                res = client.post(f"{self.ollama_url}/api/generate", json=payload)
                if res.status_code == 200:
                    return res.json().get("response", "").strip()
        except Exception:
            return None
        return None

    def parse(self, file_path: Path) -> ParsedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        np_img, b64_image = self._prepare_image(file_path)

        # 1. Run Vision Model to get semantic visual description
        vision_description = self._extract_via_ollama_vision(b64_image)

        # 2. Run OCR to extract exact text
        ocr_text = self._extract_via_ocr(np_img)

        # 3. Combine both with explicit document header
        filename = file_path.name
        sections = [f"[Image / Graphic Document: {filename}]"]

        if ocr_text:
            sections.append(f"[Extracted Text / OCR]\n{ocr_text}")
        if vision_description:
            sections.append(f"[Visual Description, Colors & Layout]\n{vision_description}")

        full_content = "\n\n".join(sections)

        with Image.open(file_path) as img:
            width, height = img.size

        return ParsedDocument(
            filename=filename,
            file_path=str(file_path.resolve()),
            file_type=file_path.suffix.lower().lstrip("."),
            text_content=full_content,
            metadata={
                "dimensions": f"{width}x{height}",
                "has_vision_description": bool(vision_description),
                "has_ocr_text": bool(ocr_text),
                "char_count": len(full_content)
            }
        )
