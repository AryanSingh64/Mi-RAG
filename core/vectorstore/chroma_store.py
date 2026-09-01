from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings
from core.chunking.text_chunker import DocumentChunk
from core.embeddings.embedder import LocalEmbedder


@dataclass
class SearchResult:
    """
    Container for search results returned from vector retrieval.
    """
    chunk_id: str
    text: str
    source_file: str
    metadata: Dict[str, Any]
    score: float  # Cosine similarity score (1.0 = exact match, 0.0 = completely unrelated)
    distance: float


class ChromaVectorStore:
    """
    ChromaDB manager for persisting and querying document chunks on disk.
    """

    def __init__(
        self,
        persist_directory: Path | str,
        collection_name: str = "rag_knowledge_base",
        embedder: Optional[LocalEmbedder] = None
    ):
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedder = embedder or LocalEmbedder()

        # Initialize persistent Chroma client (writes to SQLite on disk)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection with cosine similarity metric
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[DocumentChunk], progress_callback: Optional[Any] = None):
        """
        Embeds and stores document chunks in the vector collection in high-throughput batches.
        """
        if not chunks:
            return

        clean_metadatas = []
        for chunk in chunks:
            clean_meta = {}
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                elif isinstance(v, list):
                    clean_meta[k] = ", ".join(str(item) for item in v)
                elif v is not None:
                    clean_meta[k] = str(v)
            clean_metadatas.append(clean_meta)

        # Batch upsert in chunks of 128 for optimal vector throughput
        batch_size = 128 if getattr(self.embedder, "device", "cpu") in ("cuda", "mps") else 64
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        print(f"[*] Storing & vectorizing {len(chunks)} chunks across {total_batches} batches on {getattr(self.embedder, 'device', 'cpu').upper()}...", flush=True)

        for b_idx, i in enumerate(range(0, len(chunks), batch_size), start=1):
            b_chunks = chunks[i:i + batch_size]
            b_texts = [c.text for c in b_chunks]
            b_ids = [c.chunk_id for c in b_chunks]
            b_metas = clean_metadatas[i:i + batch_size]

            # Generate vectors using local embedder
            b_embeddings = self.embedder.embed_batch(b_texts)

            # Upsert into Chroma collection
            self.collection.upsert(
                ids=b_ids,
                embeddings=b_embeddings,
                documents=b_texts,
                metadatas=b_metas
            )

            stored = min(i + batch_size, len(chunks))
            pct = (stored / len(chunks)) * 100

            if progress_callback:
                progress_callback("indexing", stored, len(chunks), 0)

            if b_idx % 5 == 0 or b_idx == total_batches:
                print(f"  [Vector Index] Indexed {stored}/{len(chunks)} chunks ({pct:.1f}%) | Batch {b_idx}/{total_batches}", flush=True)

    def query(self, query_text: str, top_k: int = 4) -> List[SearchResult]:
        """
        Performs semantic similarity search against the indexed chunks.
        """
        query_vector = self.embedder.embed_text(query_text)

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        if results["ids"] and len(results["ids"][0]) > 0:
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                doc_text = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]

                # Chroma cosine distance ranges from 0 to 2.
                # Cosine similarity = 1.0 - distance
                similarity_score = max(0.0, 1.0 - distance)

                search_results.append(
                    SearchResult(
                        chunk_id=doc_id,
                        text=doc_text,
                        source_file=metadata.get("source_file", "unknown"),
                        metadata=metadata,
                        score=round(similarity_score, 4),
                        distance=round(distance, 4)
                    )
                )

        return search_results

    def count(self) -> int:
        """Returns total number of chunks currently indexed in the collection."""
        return self.collection.count()
