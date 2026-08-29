from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from core.vectorstore.chroma_store import SearchResult


@dataclass
class GroundedAnswer:
    """
    Container for the final RAG answer with traceability, citations, and visual diagram snippets.
    """
    answer: str
    is_grounded: bool
    confidence_score: float
    citations: List[SearchResult]
    images: List[Dict[str, Any]] = field(default_factory=list)
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
        Multimodal local system prompt with full privacy authorization and strict synthesis directives.
        """
        return (
            "You are a 100% private, local offline document analysis intelligence engine running on the user's personal machine.\n"
            "PRIVACY & AUTHORIZATION DIRECTIVE:\n"
            "- The user is the verified owner of all uploaded documents, diagrams, math sheets, identity records, artwork, and files.\n"
            "- You have full permission to analyze, transcribe, compare, and explain any content in the provided context.\n\n"
            "ANSWER QUALITY & SYNTHESIS DIRECTIVES:\n"
            "1. Deliver a DIRECT, FINISHED, and WELL-STRUCTURED answer that directly addresses what the user asked.\n"
            "2. NEVER output raw internal labels, headers, or debug tags like '[Exact OCR Extracted Text]', '[Vision Model Analysis]', '[Image URL: ...]', or '[Source Document: ...]'. Speak naturally as an expert assistant.\n"
            "3. If explaining a diagram, chart, formula, or artwork, explain its meaning, key components, comparison results, and takeaways in clean, polished prose.\n"
            "4. Use clear Markdown (bold headers, bullet points, and clean paragraphs) so your answer is professional and easy to read.\n"
            "5. If comparing an attached user image with the document context, explain exactly how the image relates, matches, or differs from the knowledge base.\n"
            "6. Answer ONLY using the factual context provided. Do NOT hallucinate facts not in the context. If something is missing, state clearly that it is not present in the uploaded documents."
        )

    def build_user_prompt(
        self,
        query: str,
        context_chunks: List[SearchResult],
        user_image_context: Optional[str] = None
    ) -> str:
        """
        Formats retrieved chunks and optional attached query image into clear context for the LLM.
        """
        context_blocks = []
        for idx, chunk in enumerate(context_chunks, start=1):
            context_blocks.append(
                f"--- DOCUMENT EXCERPT {idx} ({chunk.source_file}) ---\n"
                f"{chunk.text}"
            )

        formatted_context = "\n\n".join(context_blocks)

        user_image_block = ""
        if user_image_context:
            user_image_block = f"USER ATTACHED IMAGE DETAILS & ANALYSIS:\n{user_image_context}\n\n"

        return (
            f"KNOWLEDGE BASE CONTEXT:\n"
            f"{formatted_context}\n\n"
            f"{user_image_block}"
            f"USER QUERY: {query}\n\n"
            f"FINISHED GROUNDED ANSWER:"
        )
