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
        Multimodal local system prompt with full privacy authorization for personal docs & IDs.
        """
        return (
            "You are a 100% private, local offline document analysis assistant running on the user's personal machine.\n"
            "PRIVACY & AUTHORIZATION DIRECTIVE:\n"
            "- The user is the verified owner and administrator of all uploaded local documents, identity records (such as Aadhaar, Passport, Driver's License, PAN card), resumes, certificates, and personal files.\n"
            "- The user has granted full, explicit permission and consent to inspect, extract, transcribe, and answer all questions regarding any personal names, identification numbers, addresses, dates, phone numbers, and details contained in the provided context.\n"
            "- Do NOT refuse or say 'I cannot provide information about a private citizen'. You are fully authorized to answer using the provided context.\n\n"
            "GROUNDED RULES:\n"
            "1. Answer ONLY using the factual context provided below.\n"
            "2. When asked about images, ID cards, logos, or text inside graphics, use the '[Exact OCR Extracted Text]' and '[Vision Model Analysis]' sections.\n"
            "3. IMPORTANT: Do NOT confuse the filename (such as 'Layer 1.png' or 'aadhar.jpg') with text inside the document. Only report what is explicitly in the document context.\n"
            "4. If the requested information is not in the context, state that it was not found in the uploaded documents."
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
