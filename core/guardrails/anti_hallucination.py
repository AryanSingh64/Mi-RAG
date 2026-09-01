import re
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

    def apply_attention_reranking(
        self,
        query: str,
        chunks: List[SearchResult],
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[SearchResult]:
        """
        Attention-Guided Context Reranking.
        Computes attention focus weights by combining semantic cosine similarity,
        exact lexical term matches, and salient topics from prior conversation memory.
        """
        if not chunks:
            return []

        # Extract focus tokens from current query and recent conversation memory
        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        history_words = set()
        if history:
            for turn in history[-2:]:
                history_words.update(re.findall(r"\b\w{3,}\b", str(turn.get("content", "")).lower()))

        scored_chunks = []
        for chunk in chunks:
            chunk_lower = chunk.text.lower()
            
            # Base semantic similarity
            base_score = chunk.score

            # Lexical overlap attention boost
            query_overlap = sum(1 for w in query_words if w in chunk_lower)
            query_boost = min(0.35, (query_overlap / max(1, len(query_words))) * 0.35)

            # Conversational memory attention boost (for follow-ups)
            history_overlap = sum(1 for w in history_words if w in chunk_lower)
            history_boost = min(0.15, (history_overlap / max(1, len(history_words))) * 0.15) if history_words else 0.0

            # Diagram visual bonus if chunk contains diagram markup
            diag_boost = 0.05 if "[DIAGRAM" in chunk.text else 0.0

            # Final attention-weighted score
            attention_score = base_score + query_boost + history_boost + diag_boost
            scored_chunks.append((attention_score, chunk))

        # Sort descending by attention score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks]

    def build_grounded_system_prompt(self) -> str:
        """
        Multimodal local system prompt with full privacy authorization, conversation memory directives,
        and strict anti-hallucination synthesis.
        """
        return (
            "You are a 100% private, local offline document analysis intelligence engine running on the user's personal machine.\n"
            "PRIVACY & AUTHORIZATION DIRECTIVE:\n"
            "- The user is the verified owner of all uploaded documents, diagrams, math sheets, identity records, artwork, and files.\n"
            "- You have full permission to analyze, transcribe, compare, and explain any content in the provided context.\n\n"
            "ANSWER QUALITY & CONVERSATIONAL MEMORY DIRECTIVES:\n"
            "1. Deliver a DIRECT, FINISHED, and WELL-STRUCTURED answer that directly addresses what the user asked.\n"
            "2. CONVERSATION MEMORY: Use prior conversation turns to resolve pronouns (e.g. 'it', 'this', 'that', 'they', 'the previous method'). Maintain smooth conversational continuity.\n"
            "3. NEVER output raw internal labels, headers, or debug tags like '[Exact OCR Extracted Text]', '[Vision Model Analysis]', '[Image URL: ...]', or '[Source Document: ...]'. Speak naturally as an expert assistant.\n"
            "4. If explaining a diagram, chart, formula, or artwork, explain its meaning, key components, comparison results, and takeaways in clean, polished prose.\n"
            "5. Use clear Markdown (bold headers, bullet points, and clean paragraphs) so your answer is professional and easy to read.\n"
            "6. Answer ONLY using the factual context provided. Do NOT hallucinate facts not in the context. If something is missing, state clearly that it is not present in the uploaded documents.\n"
            "7. MATHEMATICAL & SCIENTIFIC SYMBOLS: Use clean LaTeX math delimiters for formulas, tolerances, and scientific quantities (e.g. `$1.25 \\pm 0.80$ cm`, `$\\times$`, `$\\approx$`, `$\\le$`, `$\\ge$`, `$\\alpha$`, `$\\beta$`, `$$ E = mc^2 $$`) so math renders crisply."
        )

    def build_user_prompt(
        self,
        query: str,
        context_chunks: List[SearchResult],
        user_image_context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        feedback_exemplars: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Formats retrieved chunks, conversation memory, feedback exemplars, and optional attached query image into clear context for the LLM.
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

        history_block = ""
        if conversation_history:
            turns = []
            for h in conversation_history[-4:]:
                role = "User" if h.get("role") == "user" else "Assistant"
                content = str(h.get("content", "")).strip()
                turns.append(f"{role}: {content}")
            if turns:
                history_block = "RECENT CONVERSATION MEMORY (PRIOR DIALOGUE TURNS):\n" + "\n".join(turns) + "\n\n"

        exemplar_block = ""
        if feedback_exemplars:
            ex_items = []
            for ex in feedback_exemplars[:2]:
                q = ex.get("query", "").strip()
                a = ex.get("answer", "").strip()
                if q and a:
                    ex_items.append(f"Query: {q}\nApproved Answer: {a}")
            if ex_items:
                exemplar_block = "USER-VERIFIED EXEMPLARY RESPONSES (FORMATTING & ACCURACY REFERENCE):\n" + "\n\n".join(ex_items) + "\n\n"

        return (
            f"KNOWLEDGE BASE CONTEXT:\n"
            f"{formatted_context}\n\n"
            f"{user_image_block}"
            f"{history_block}"
            f"{exemplar_block}"
            f"USER QUERY: {query}\n\n"
            f"FINISHED GROUNDED ANSWER:"
        )
