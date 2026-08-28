from dataclasses import dataclass
from typing import List, Optional
from core.vectorstore.chroma_store import SearchResult


@dataclass
class GroundedAnswer:
    """
    Container for the final RAG answer with traceability and citations.
    """
    answer: str
    is_grounded: bool
    confidence_score: float
    citations: List[SearchResult]
    warning: Optional[str] = None


class AntiHallucinationEngine:
    """
    Guardrails engine that filters retrieved context, constructs clean prompts,
    and enforces citation-backed responses.
    """

    def __init__(self, min_similarity_threshold: float = 0.20):
        self.min_similarity_threshold = min_similarity_threshold

    def filter_relevant_chunks(self, search_results: List[SearchResult]) -> List[SearchResult]:
        """
        Discards retrieved chunks that fall below the minimum cosine similarity threshold.
        """
        return [r for r in search_results if r.score >= self.min_similarity_threshold]

    def build_grounded_system_prompt(self) -> str:
        """
        Clean, direct system prompt optimized for fast local models.
        """
        return (
            "You are a precise enterprise assistant. "
            "Read the provided context and answer the user's question directly and concisely using only facts from the context."
        )

    def build_user_prompt(self, query: str, context_chunks: List[SearchResult]) -> str:
        """
        Formats retrieved chunks into clear, structured context for the LLM.
        """
        context_blocks = []
        for idx, chunk in enumerate(context_chunks, start=1):
            context_blocks.append(
                f"[Document: {chunk.source_file}]\n"
                f"{chunk.text}"
            )

        formatted_context = "\n\n".join(context_blocks)

        return (
            f"Context:\n"
            f"{formatted_context}\n\n"
            f"Question: {query}\n\n"
            f"Answer based only on the context:"
        )
