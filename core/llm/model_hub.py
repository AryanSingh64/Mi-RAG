from typing import Any, Dict, List
import httpx
from core.system.hardware_detector import HardwareDetector


# Curated catalog of trending and top-performing Ollama models
OLLAMA_MODEL_CATALOG: List[Dict[str, Any]] = [
    # Vision & Multimodal Models
    {
        "name": "moondream:latest",
        "tag": "moondream:latest",
        "display_name": "Moondream2 (Vision & OCR)",
        "category": "vision",
        "size_gb": 1.7,
        "required_vram_gb": 2.0,
        "description": "Ultra-fast lightweight vision model. Extracts text, logos, colors, and layout in ~1s.",
        "best_for": "Logos, Diagrams & Fast OCR"
    },
    {
        "name": "deepseek-ocr:latest",
        "tag": "deepseek-ocr:latest",
        "display_name": "DeepSeek-OCR (Vision)",
        "category": "vision",
        "size_gb": 2.2,
        "required_vram_gb": 2.5,
        "description": "High-accuracy token-efficient vision language model specializing in document OCR.",
        "best_for": "Documents, Invoices & Receipts"
    },
    {
        "name": "qwen2.5vl:3b",
        "tag": "qwen2.5vl:3b",
        "display_name": "Qwen 2.5-VL 3B (Vision)",
        "category": "vision",
        "size_gb": 2.5,
        "required_vram_gb": 3.0,
        "description": "Flagship visual language model with spatial understanding and fine document parsing.",
        "best_for": "Charts, Complex Graphics & Tables"
    },
    {
        "name": "granite3.2-vision:2b",
        "tag": "granite3.2-vision:2b",
        "display_name": "IBM Granite 3.2 Vision 2B",
        "category": "vision",
        "size_gb": 1.8,
        "required_vram_gb": 2.2,
        "description": "IBM compact enterprise vision model optimized for infographic and chart extraction.",
        "best_for": "Enterprise Documents & Infographics"
    },
    {
        "name": "llava:7b",
        "tag": "llava:7b",
        "display_name": "LLaVA 1.6 7B (Vision)",
        "category": "vision",
        "size_gb": 4.5,
        "required_vram_gb": 5.5,
        "description": "End-to-end multimodal model combining Vicuna with visual encoder.",
        "best_for": "Detailed Visual Reasoning"
    },
    {
        "name": "llama3.2-vision:11b",
        "tag": "llama3.2-vision:11b",
        "display_name": "Llama 3.2 Vision 11B",
        "category": "vision",
        "size_gb": 7.8,
        "required_vram_gb": 8.5,
        "description": "Meta's flagship multimodal model for complex image reasoning.",
        "best_for": "High-End Visual Analytics"
    },

    # Text & Reasoning LLMs
    {
        "name": "llama3.2:3b",
        "tag": "llama3.2:3b",
        "display_name": "Llama 3.2 3B (Text)",
        "category": "text",
        "size_gb": 2.0,
        "required_vram_gb": 2.5,
        "description": "Meta's high-efficiency compact LLM. Excellent instruction following and reasoning.",
        "best_for": "General RAG & Question Answering"
    },
    {
        "name": "llama3.2:1b",
        "tag": "llama3.2:1b",
        "display_name": "Llama 3.2 1B (Text)",
        "category": "text",
        "size_gb": 1.3,
        "required_vram_gb": 1.5,
        "description": "Blazing-fast ultra-lightweight LLM. Sub-50ms token generation on GPU.",
        "best_for": "Real-Time Chat & Low Memory"
    },
    {
        "name": "qwen2.5:3b",
        "tag": "qwen2.5:3b",
        "display_name": "Qwen 2.5 3B (Text & Code)",
        "category": "text",
        "size_gb": 2.2,
        "required_vram_gb": 2.8,
        "description": "Powerful bilingual model with exceptional coding and structured extraction capabilities.",
        "best_for": "Technical Docs & Code RAG"
    },
    {
        "name": "gemma2:2b",
        "tag": "gemma2:2b",
        "display_name": "Google Gemma 2 2B",
        "category": "text",
        "size_gb": 1.6,
        "required_vram_gb": 2.0,
        "description": "Google DeepMind's compact model built with lightweight architecture.",
        "best_for": "Summarization & Concise Output"
    },
    {
        "name": "phi3.5:3.8b",
        "tag": "phi3.5:3.8b",
        "display_name": "Microsoft Phi-3.5 3.8B",
        "category": "text",
        "size_gb": 2.2,
        "required_vram_gb": 3.0,
        "description": "Microsoft's state-of-the-art small language model with high logical reasoning.",
        "best_for": "Complex Logic & Synthesis"
    },
    {
        "name": "mistral:7b",
        "tag": "mistral:7b",
        "display_name": "Mistral 7B Instruct",
        "category": "text",
        "size_gb": 4.1,
        "required_vram_gb": 5.5,
        "description": "Industry benchmark 7B instruction model with rich language mastery.",
        "best_for": "Long-form Synthesis"
    }
]


class ModelHub:
    """
    Ollama Model Catalog and Downloader with hardware-aware recommendation scoring.
    """

    @staticmethod
    def get_installed_models(ollama_url: str = "http://localhost:11434") -> List[str]:
        try:
            with httpx.Client(timeout=4.0) as client:
                res = client.get(f"{ollama_url}/api/tags")
                if res.status_code == 200:
                    return [m.get("name") for m in res.json().get("models", [])]
        except Exception:
            return []
        return []

    @classmethod
    def get_catalog_with_status(cls, ollama_url: str = "http://localhost:11434") -> Dict[str, Any]:
        installed = cls.get_installed_models(ollama_url)
        specs = HardwareDetector.get_specs()

        models_with_status = []
        for item in OLLAMA_MODEL_CATALOG:
            # Check if installed (matching name or tag or prefix)
            is_installed = any(
                item["name"] == inst or
                item["tag"] == inst or
                item["name"].split(":")[0] == inst.split(":")[0]
                for inst in installed
            )

            compat = HardwareDetector.evaluate_model_compatibility(item["required_vram_gb"])

            model_data = {
                **item,
                "is_installed": is_installed,
                "compatibility": compat,
                "is_recommended": compat["status"] in ["recommended", "compatible"] and item["size_gb"] <= 3.5
            }
            models_with_status.append(model_data)

        # Also append any custom installed models not in catalog
        catalog_names = [m["name"].split(":")[0] for m in OLLAMA_MODEL_CATALOG]
        for inst in installed:
            inst_base = inst.split(":")[0]
            if inst_base not in catalog_names and inst not in [m["name"] for m in models_with_status]:
                is_vis = any(vk in inst.lower() for vk in ["vision", "moondream", "llava", "ocr", "vl"])
                models_with_status.append({
                    "name": inst,
                    "tag": inst,
                    "display_name": f"{inst} (Custom Local)",
                    "category": "vision" if is_vis else "text",
                    "size_gb": 2.0,
                    "required_vram_gb": 2.5,
                    "description": "Locally installed custom model.",
                    "best_for": "Local Workloads",
                    "is_installed": True,
                    "compatibility": HardwareDetector.evaluate_model_compatibility(2.5),
                    "is_recommended": True
                })

        return {
            "specs": specs,
            "installed_count": len(installed),
            "catalog": models_with_status
        }
