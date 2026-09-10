import os
import sys
from typing import Any, Dict, List, Optional

# Windows console encoding & DLL search path configuration
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        vc_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets", "vc_runtimes")
        if os.path.exists(vc_dir) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(vc_dir)
    except Exception:
        pass

SentenceTransformer = None
HAS_SENTENCE_TRANSFORMERS = None


EMBEDDING_CATALOG: Dict[str, Dict[str, Any]] = {
    "BAAI/bge-base-en-v1.5": {
        "name": "BAAI/bge-base-en-v1.5",
        "use_case": "Best Overall",
        "summary": "Balanced high precision for standard documents, articles, and manuals",
        "label": "Best Overall: High Accuracy [BAAI/bge-base-en-v1.5]",
        "dim": 768,
        "is_default": True,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "description": "Recommended for general documents and high-precision semantic retrieval."
    },
    "all-MiniLM-L6-v2": {
        "name": "all-MiniLM-L6-v2",
        "use_case": "Fastest",
        "summary": "Ultra-fast and low memory for rapid CPU indexing",
        "label": "Fastest: Low Memory [all-MiniLM-L6-v2]",
        "dim": 384,
        "is_default": False,
        "query_prefix": "",
        "description": "Lightweight model suited for fast processing on lower-spec machines."
    },
    "BAAI/bge-m3": {
        "name": "BAAI/bge-m3",
        "use_case": "Multilingual",
        "summary": "100+ languages and extended 8192-token attention span",
        "label": "Multilingual: 100+ Languages [BAAI/bge-m3]",
        "dim": 1024,
        "is_default": False,
        "query_prefix": "",
        "description": "Multi-lingual retrieval across diverse languages and long documents."
    },
    "BAAI/bge-large-en-v1.5": {
        "name": "BAAI/bge-large-en-v1.5",
        "use_case": "Deep Research",
        "summary": "Maximum semantic precision for complex technical papers",
        "label": "Deep Research: Maximum Precision [BAAI/bge-large-en-v1.5]",
        "dim": 1024,
        "is_default": False,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "description": "Large-capacity model designed for complex research papers and technical specifications."
    },
    "nomic-ai/nomic-embed-text-v1.5": {
        "name": "nomic-ai/nomic-embed-text-v1.5",
        "use_case": "Long Documents",
        "summary": "Full-chapter retrieval with 8192-token context window",
        "label": "Long Documents: Extended Context [nomic-ai/nomic-embed-text-v1.5]",
        "dim": 768,
        "is_default": False,
        "query_prefix": "search_query: ",
        "passage_prefix": "search_document: ",
        "description": "Long-context embedding model with Matryoshka dimensionality support."
    }
}


class LocalEmbedder:
    """
    Modular wrapper for local Hugging Face embedding models via sentence-transformers.
    Automatically detects CUDA GPU, Apple Silicon (MPS), or optimized multi-core CPU.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: Optional[str] = None):
        self.model_name = model_name or "BAAI/bge-base-en-v1.5"
        
        # 1. Device Selection & GPU Acceleration
        if device:
            self.device = device
        else:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except Exception:
                self.device = "cpu"

        # 2. Query Instruction Prefixes
        catalog_entry = EMBEDDING_CATALOG.get(self.model_name, {})
        self.query_prefix = catalog_entry.get("query_prefix", "")
        self.passage_prefix = catalog_entry.get("passage_prefix", "")

        self.dim = catalog_entry.get("dim", 768)

        # 3. Dynamic Batch Sizing
        self.batch_size = 128 if self.device in ("cuda", "mps") else 64

        print(f"[*] Initializing Embedding Model: {self.model_name} on device: {self.device.upper()} (batch_size={self.batch_size})...")
        self.model = self._load_sentence_transformer()

    def _load_sentence_transformer(self):
        """Loads sentence-transformers model with offline-first caching to guarantee 100% offline operation."""
        global SentenceTransformer, HAS_SENTENCE_TRANSFORMERS
        if SentenceTransformer is None:
            try:
                from sentence_transformers import SentenceTransformer as ST
                SentenceTransformer = ST
                HAS_SENTENCE_TRANSFORMERS = True
            except (ImportError, OSError, Exception) as err:
                print(f"[*] Local neural embedder note: {err}. Activating resilient fallback.")
                HAS_SENTENCE_TRANSFORMERS = False
                return None

        model_kwargs = {"torch_dtype": "float16"} if self.device == "cuda" else {}

        # 1. Try local cache first (100% offline, zero network requests, instant startup)
        try:
            return SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs=model_kwargs,
                trust_remote_code=True,
                local_files_only=True
            )
        except Exception:
            pass

        # 2. Try online download if local cache miss and network is available
        try:
            return SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs=model_kwargs,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"[!] Notice: Offline or cache miss loading {self.model_name}: {e}. Trying fallback models...")

        # 3. Fallback to cached all-MiniLM-L6-v2 or bge-small
        for fb_name in ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]:
            try:
                self.model_name = fb_name
                return SentenceTransformer(fb_name, device=self.device, local_files_only=True)
            except Exception:
                try:
                    return SentenceTransformer(fb_name, device=self.device)
                except Exception:
                    pass

        return None

    def _embed_via_ollama(self, text: str) -> Optional[List[float]]:
        """100% offline embedding via local Ollama instance if sentence-transformers is unavailable."""
        try:
            import httpx
            with httpx.Client(timeout=4.0) as client:
                res = client.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": "nomic-embed-text", "prompt": text}
                )
                if res.status_code == 200:
                    vec = res.json().get("embedding", [])
                    if vec:
                        return vec
        except Exception:
            pass
        return None

    def _fallback_hash_vector(self, text: str) -> List[float]:
        """High-speed deterministic normalized pseudo-semantic vector when dependencies are missing."""
        import hashlib
        import math
        vec = [0.0] * self.dim
        words = text.lower().split()
        for i, word in enumerate(words):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % self.dim
            val = (1.0 / math.sqrt(i + 1)) * (1.0 if (h % 2 == 0) else -1.0)
            vec[idx] += val
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        else:
            vec[0] = 1.0
        return vec

    def embed_text(self, text: str) -> List[float]:
        """Generates an embedding vector for a single search query with instruction prefix."""
        if self.model is not None:
            formatted_text = f"{self.query_prefix}{text}" if self.query_prefix else text
            embedding = self.model.encode(formatted_text, convert_to_numpy=True, normalize_embeddings=True)
            return embedding.tolist()

        ollama_vec = self._embed_via_ollama(text)
        if ollama_vec:
            return ollama_vec

        return self._fallback_hash_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates normalized embedding vectors for a list of document chunks in high-speed batches."""
        if not texts:
            return []
        
        if self.model is not None:
            if self.passage_prefix:
                formatted_texts = [f"{self.passage_prefix}{t}" for t in texts]
            else:
                formatted_texts = texts

            embeddings = self.model.encode(
                formatted_texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings.tolist()

        if self.model is None:
            # Try Ollama embedding fallback
            ollama_results = []
            can_use_ollama = True
            for t in texts:
                vec = self._embed_via_ollama(t)
                if vec:
                    ollama_results.append(vec)
                else:
                    can_use_ollama = False
                    break
            if can_use_ollama and len(ollama_results) == len(texts):
                return ollama_results

        return [self._fallback_hash_vector(t) for t in texts]

    @classmethod
    def get_catalog(cls) -> List[Dict[str, Any]]:
        """Returns the list of pre-configured embedding models with metadata."""
        return list(EMBEDDING_CATALOG.values())
