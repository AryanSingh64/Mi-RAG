from typing import List
from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """
    Wrapper for local Hugging Face embedding models via sentence-transformers.
    Runs entirely offline on CPU/GPU with zero cloud API dependencies.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        print(f"📦 Loading embedding model: {model_name} (runs on CPU/local)...")
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        """Generates an embedding vector for a single search query."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a list of document chunks in batches."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()
