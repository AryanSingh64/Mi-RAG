import base64
import concurrent.futures
import io
import os
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
import httpx
from core.ingestion.base import BaseDocumentParser, ParsedDocument


class VisionImageParser(BaseDocumentParser):
    """
    Multimodal Multi-Model Image Parser.
    Extracts text using RapidOCR (with alpha channel flattening) AND generates rich semantic descriptions
    across multiple selected Ollama Vision models (Ensemble Fusion).
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        vision_models: Optional[Union[List[str], str]] = None,
        output_images_dir: Optional[Path] = None,
        session_id: Optional[str] = None
    ):
        self.ocr_engine = RapidOCR()
        self.ollama_url = ollama_url
        self.output_images_dir = Path(output_images_dir) if output_images_dir else None
        if self.output_images_dir:
            self.output_images_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id

        if isinstance(vision_models, str):
            self.vision_models = [m.strip() for m in vision_models.split(",") if m.strip()]
        elif isinstance(vision_models, list) and vision_models:
            self.vision_models = vision_models
        else:
            self.vision_models = ["moondream"]

    def _resolve_vision_model(self, target_model: str) -> str:
        """Finds the exact matching vision model tag in local Ollama."""
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.ollama_url}/api/tags")
                if res.status_code == 200:
                    models = [m.get("name") for m in res.json().get("models", [])]
                    if target_model in models:
                        return target_model
                    if f"{target_model}:latest" in models:
                        return f"{target_model}:latest"
                    for m in models:
                        if target_model.lower() in m.lower():
                            return m
        except Exception:
            pass
        return target_model

    def _prepare_image(self, image_path: Path) -> tuple[np.ndarray, str]:
        """
        Loads image, normalizes alpha transparency, keeps native resolution for OCR,
        and generates an optimized 768px base64 thumbnail for ultra-fast vision LLM processing.
        """
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgba = img.convert("RGBA")
                bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                bg.paste(rgba, (0, 0), rgba)
                rgb_img = bg.convert("RGB")
            else:
                rgb_img = img.convert("RGB")

            # Native resolution for RapidOCR
            np_img = np.array(rgb_img)

            # High-efficiency thumbnail (max 768px) for vision model speed
            vis_img = rgb_img.copy()
            vis_img.thumbnail((768, 768), Image.Resampling.LANCZOS)

            buffered = io.BytesIO()
            vis_img.save(buffered, format="JPEG", quality=85, optimize=True)
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
        except Exception as e:
            print(f"[*] OCR extraction note: {e}")
            return ""

    def _extract_from_single_vision_model(self, model_name: str, b64_image: str) -> Optional[str]:
        """Queries a single vision model with persistent VRAM caching and fast bounded token generation."""
        resolved_model = self._resolve_vision_model(model_name)
        print(f"[*] Querying Vision Model: {resolved_model}...")

        prompt = (
            "Analyze this image: "
            "1. Transcribe all text, numbers, brands, or words exactly as written. "
            "2. Describe colors, visual layout, and graphic style. "
            "3. State what this image represents."
        )

        payload = {
            "model": resolved_model,
            "prompt": prompt,
            "images": [b64_image],
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "num_gpu": 99,
                "num_thread": os.cpu_count() or 12,
                "num_predict": 250,
                "temperature": 0.1
            }
        }

        try:
            with httpx.Client(timeout=240.0) as client:
                res = client.post(f"{self.ollama_url}/api/generate", json=payload)
                if res.status_code == 200:
                    description = res.json().get("response", "").strip()
                    if len(description) > 5:
                        print(f"[OK] {resolved_model} produced description ({len(description)} chars)")
                        return description
                    else:
                        print(f"[*] {resolved_model} produced no text. Using OCR extraction.")
                        return None
                else:
                    try:
                        err_msg = res.json().get("error", res.text)
                    except Exception:
                        err_msg = res.text
                    if "unknown model architecture" in err_msg:
                        print(f"[*] Note: {resolved_model} requires Ollama mllama update. Using complementary vision models.")
                    else:
                        print(f"[*] Vision model note ({resolved_model}): {err_msg[:80]}")
                    return None
        except Exception as e:
            print(f"[*] Vision model note ({resolved_model}): {e}")
            return None
        return None

    def _extract_all_vision_descriptions(self, b64_image: str) -> List[tuple[str, str]]:
        """Runs all selected vision models in parallel for minimum latency."""
        if not self.vision_models:
            return []

        results = []
        import concurrent.futures

        def _worker(model):
            desc = self._extract_from_single_vision_model(model, b64_image)
            return (model, desc) if desc else None

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(self.vision_models))) as executor:
            futures = [executor.submit(_worker, m) for m in self.vision_models]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                except Exception as e:
                    print(f"[!] Parallel vision worker error: {e}")

        return results

    def parse(self, file_path: Path) -> ParsedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        np_img, b64_image = self._prepare_image(file_path)

        # 1. Run all selected Vision Models
        vision_results = self._extract_all_vision_descriptions(b64_image)

        # 2. Run OCR
        ocr_text = self._extract_via_ocr(np_img)

        # 3. Format combined ensemble sections
        filename = file_path.name
        img_url = f"/api/sessions/{self.session_id}/images/{filename}" if self.session_id else f"/images/{filename}"

        if self.output_images_dir:
            target_path = self.output_images_dir / filename
            try:
                if not target_path.exists():
                    import shutil
                    shutil.copy2(file_path, target_path)
            except Exception as e:
                print(f"[*] Note copying image to session gallery: {e}")

        sections = [
            f"Source File Name: {filename}",
            f"[Image URL: {img_url}]"
        ]

        if ocr_text:
            sections.append(f"[Exact OCR Extracted Text]\n{ocr_text}")

        for model_name, desc in vision_results:
            sections.append(f"[Vision Model Analysis ({model_name})]\n{desc}")

        if not ocr_text and not vision_results:
            sections.append("[Visual Content]\nNo readable text or visual description could be generated for this image.")

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
                "vision_models_used": [m for m, _ in vision_results],
                "has_ocr_text": bool(ocr_text),
                "char_count": len(full_content)
            }
        )

    def describe_and_ocr_image(self, image_path: Path | str) -> dict:
        """
        Extracts OCR text and multimodal vision descriptions for a query/search image.
        Returns a dict with 'ocr_text', 'description', and 'combined_summary'.
        """
        path = Path(image_path)
        if not path.exists():
            return {"ocr_text": "", "description": "", "combined_summary": ""}

        np_img, b64_img = self._prepare_image(path)
        ocr_text = self._extract_via_ocr(np_img)

        descriptions = []
        if self.vision_models:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(self.vision_models), 2)) as executor:
                future_to_model = {
                    executor.submit(self._extract_from_single_vision_model, model, b64_img): model
                    for model in self.vision_models
                }
                for future in concurrent.futures.as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        desc = future.result()
                        if desc:
                            descriptions.append(f"[{model_name}]: {desc}")
                    except Exception as e:
                        print(f"[!] Vision model {model_name} error: {e}")

        combined_desc = "\n".join(descriptions)
        summary_parts = []
        if ocr_text:
            summary_parts.append(f"Text detected in image:\n{ocr_text}")
        if combined_desc:
            summary_parts.append(f"Visual Analysis & Details:\n{combined_desc}")

        combined_summary = "\n\n".join(summary_parts) if summary_parts else "Image provided with no detectable text or description."

        return {
            "ocr_text": ocr_text,
            "description": combined_desc,
            "combined_summary": combined_summary
        }
