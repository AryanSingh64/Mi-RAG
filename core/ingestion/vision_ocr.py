import base64
from pathlib import Path
from typing import Optional
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
import httpx
from core.ingestion.base import BaseDocumentParser, ParsedDocument


class VisionImageParser(BaseDocumentParser):
    """
    Multimodal Image Parser.
    Extracts text using RapidOCR AND generates a rich semantic description via Ollama Vision.
    """

    def __init__(self, ollama_url: str = "http://localhost:11434", vision_model: str = "moondream"):
        self.ocr_engine = RapidOCR()
        self.ollama_url = ollama_url
        self.vision_model = vision_model

    def _extract_via_ocr(self, image_path: Path) -> str:
        """Fast local CPU OCR extraction."""
        result, _ = self.ocr_engine(str(image_path))
        if not result:
            return ""
        lines = [item[1] for item in result if item[1].strip()]
        return "\n".join(lines)

    def _extract_via_ollama_vision(self, image_path: Path) -> Optional[str]:
        """Generates detailed semantic description of image contents using Ollama Vision."""
        try:
            with open(image_path, "rb") as img_file:
                b64_image = base64.b64encode(img_file.read()).decode("utf-8")

            prompt = (
                "Describe this image in comprehensive detail. "
                "Explain what it is, its main subject, all objects, concepts, diagrams, charts, or scenes depicted. "
                "If it contains visual data, explain the structure and meaning."
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

        # 1. Run Vision Model to get semantic description
        vision_description = self._extract_via_ollama_vision(file_path)

        # 2. Run OCR to extract exact text
        ocr_text = self._extract_via_ocr(file_path)

        # 3. Combine both
        sections = []
        if vision_description:
            sections.append(f"[Visual Description & Summary]\n{vision_description}")
        if ocr_text:
            sections.append(f"[Extracted Text / Labels]\n{ocr_text}")

        full_content = "\n\n".join(sections) if sections else "Image containing no readable text or visual description."

        with Image.open(file_path) as img:
            width, height = img.size

        return ParsedDocument(
            filename=file_path.name,
            file_path=str(file_path.resolve()),
            file_type=file_path.suffix.lower().lstrip("."),
            text_content=full_content,
            metadata={
                "dimensions": f"{width}x{height}",
                "has_vision_description": bool(vision_description),
                "char_count": len(full_content)
            }
        )
