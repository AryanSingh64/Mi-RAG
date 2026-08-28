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

    def __init__(self, min_similarity_threshold: float = 0.05):
        self.min_similarity_threshold = min_similarity_threshold

    def filter_relevant_chunks(self, search_results: List[SearchResult]) -> List[SearchResult]:
        """
        Discards retrieved chunks that fall below the minimum cosine similarity threshold.
        """
        return [r for r in search_results if r.score >= self.min_similarity_threshold]

    def build_grounded_system_prompt(self) -> str:
        """
        Multimodal system prompt that distinguishes filenames from actual image contents.
        """
        return (
            "You are a strict, grounded enterprise assistant with multimodal visual capabilities.\n"
            "Rules:\n"
            "1. Answer ONLY using the factual context provided below.\n"
            "2. When asked about images, logos, colors, fonts, or text inside graphics, use the '[Exact OCR Extracted Text]' and '[Visual Description, Logo Content, Colors & Fonts]' sections.\n"
            "3. IMPORTANT: Do NOT confuse the filename or file title (such as 'Layer 1.png') with the text, words, or brand inside the image. Only report what is explicitly described in the visual description or OCR.\n"
            "4. If the context does not contain the answer, state that it was not found in the documentation."
        )

    def build_user_prompt(self, query: str, context_chunks: List[SearchResult]) -> str:
        """
        Formats retrieved chunks into clear, structured context for the LLM.
        """
        context_blocks = []
        for idx, chunk in enumerate(context_chunks, start=1):
            context_blocks.append(
                f"[Source Document: {chunk.source_file}]\n"
                f"{chunk.text}"
            )

        formatted_context = "\n\n".join(context_blocks)

        return (
            f"Context:\n"
            f"{formatted_context}\n\n"
            f"Question: {query}\n\n"
            f"Grounded Answer:"
        )
