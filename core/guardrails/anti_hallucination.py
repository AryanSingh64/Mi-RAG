import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
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
    Guardrails engine that filters retrieved context, evaluates answer grounding,
    detects negative/out-of-domain fallbacks, and enforces citation-backed responses.
    """

    def __init__(self, min_similarity_threshold: float = 0.35):
        self.min_similarity_threshold = min_similarity_threshold

    def filter_relevant_chunks(self, search_results: List[SearchResult]) -> List[SearchResult]:
        """
        Discards retrieved chunks that fall below the calibrated cosine similarity threshold.
        """
        return [r for r in search_results if r.score >= self.min_similarity_threshold]

    def calibrate_confidence(self, raw_score: float, has_lexical_match: bool = False) -> float:
        """
        Calibrates raw high-dimensional cosine similarity (typically 0.30 - 0.88)
        into a realistic 0.0 to 1.0 confidence score.
        """
        # Noise floor is ~0.38 for general sentence embeddings
        if raw_score < 0.38 and not has_lexical_match:
            return 0.0
        
        # Smooth sigmoid / linear mapping from [0.38, 0.85] -> [0.10, 0.98]
        calibrated = (raw_score - 0.38) / (0.85 - 0.38)
        calibrated = max(0.0, min(1.0, calibrated))
        
        if has_lexical_match:
            calibrated = min(1.0, calibrated + 0.15)
            
        return round(calibrated, 4)

    def evaluate_grounding(
        self,
        query: str,
        answer: str,
        relevant_chunks: List[SearchResult]
    ) -> Tuple[bool, float, List[SearchResult]]:
        """
        Evaluates whether the generated response is genuinely grounded in the documents.
        Detects out-of-domain answers ('no mention', 'not found', 'random string') and
        strips citations/images so false positives are eliminated.
        """
        if not relevant_chunks or not answer:
            return False, 0.0, []

        ans_lower = answer.lower()
        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        
        # Common stop-words to ignore for lexical check
        stop_words = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "this", "that", "these", "those", "is", "are", "was", "were", "give", "tell", "show", "find"}
        content_query_words = query_words - stop_words

        # Check for explicit negative statements indicating absence of evidence in documents
        negative_patterns = [
            r"\bno (?:mention|information|reference|data|evidence|record|detail)\b",
            r"\bnot (?:mentioned|found|present|discussed|referenced|available|included|stated)\b",
            r"\bcannot find\b",
            r"\bcould not find\b",
            r"\bdoes not (?:mention|contain|provide|discuss|reference|include)\b",
            r"\bno direct evidence\b",
            r"\bappears to be a random (?:string|character|sequence)\b",
            r"\bis not present in the (?:uploaded|provided) documents\b",
            r"\bthere is no (?:mention|record|information)\b",
            r"\bbased on the provided documents, there is no\b",
            r"\bi could not find any information\b"
        ]

        is_negative_response = any(re.search(pat, ans_lower) for pat in negative_patterns)
        
        if is_negative_response:
            # Query was asked about something NOT in the documents -> Grounded = False, no citations
            return False, 0.0, []

        # Check lexical match across all retrieved chunks
        all_chunk_text = " ".join(c.text.lower() for c in relevant_chunks)
        lexical_matches = sum(1 for w in content_query_words if w in all_chunk_text) if content_query_words else 1
        has_lexical = (lexical_matches > 0)

        # Average similarity of top 3 chunks
        top_chunks = relevant_chunks[:3]
        raw_avg = sum(c.score for c in top_chunks) / len(top_chunks) if top_chunks else 0.0
        calibrated_score = self.calibrate_confidence(raw_avg, has_lexical)

        # If semantic score is too low or no lexical keywords match for specific query
        if calibrated_score < 0.25 and not has_lexical and len(content_query_words) > 0:
            return False, calibrated_score, []

        return True, calibrated_score, relevant_chunks

    def apply_attention_reranking(
        self,
        query: str,
        chunks: List[SearchResult],
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[SearchResult]:
        """
        Attention-Guided Context Reranking.
        Combines semantic similarity, lexical overlap, and conversation focus.
        """
        if not chunks:
            return []

        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        history_words = set()
        if history:
            for turn in history[-2:]:
                history_words.update(re.findall(r"\b\w{3,}\b", str(turn.get("content", "")).lower()))

        scored_chunks = []
        for chunk in chunks:
            chunk_lower = chunk.text.lower()
            base_score = chunk.score

            # Lexical overlap attention boost
            query_overlap = sum(1 for w in query_words if w in chunk_lower)
            query_boost = min(0.30, (query_overlap / max(1, len(query_words))) * 0.30)

            # Conversational memory attention boost
            history_overlap = sum(1 for w in history_words if w in chunk_lower)
            history_boost = min(0.12, (history_overlap / max(1, len(history_words))) * 0.12) if history_words else 0.0

            # Diagram visual bonus if chunk contains diagram markup
            diag_boost = 0.05 if "[DIAGRAM" in chunk.text else 0.0

            attention_score = base_score + query_boost + history_boost + diag_boost
            scored_chunks.append((attention_score, chunk))

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
            "3. NEVER output raw internal labels, headers, debug tags, or server URLs (e.g. '[Exact OCR Extracted Text]', '[Vision Model Analysis]', or '/api/sessions/...'). Speak naturally as an expert assistant.\n"
            "4. If explaining a diagram, chart, formula, or artwork, explain its meaning, key components, comparison results, and takeaways in clean, polished prose.\n"
            "5. Use clear Markdown (bold headers, bullet points, and clean paragraphs) so your answer is professional and easy to read.\n"
            "6. Answer ONLY using the factual context provided. Do NOT hallucinate facts not in the context. If something is missing, state clearly that it is not present in the uploaded documents.\n"
            "7. MATHEMATICAL & SCIENTIFIC SYMBOLS: Use clean LaTeX math delimiters for formulas, tolerances, and scientific quantities (e.g. `$1.25 \\pm 0.80$ cm`, `$\\times$`, `$\\approx$`, `$\\le$`, `$\\ge$`, `$\\alpha$`, `$\\beta$`, `$$ E = mc^2 $$`) so math renders crisply.\n"
            "8. VISUAL MEDIA & FIGURES: When discussing figures, diagrams, or visual documents from the context, describe their visual contents, key features, and findings directly. Figures and diagrams are automatically displayed in the interactive gallery below your answer, so NEVER say 'I cannot display images' and NEVER output raw server file paths or URLs.\n"
            "9. ANTI-FABRICATION FOR SPARSE/CALENDAR/IMAGE PAGES: If the retrieved document excerpts consist of calendar dates, numbers, sparse words, or visual photos, DO NOT invent or fabricate fictional stories, cartoon characters, animals in clothes, or unmentioned personas. If the user asks 'what is this' or 'what is in this document', state accurately that it is a calendar / visual photo collection and describe the verified textual or visual elements.\n"
            "10. VISUAL COMPOSITION, COLORS & LOGICAL AESTHETICS: When the user asks what an image represents, what is special in it, or asks about composition, colors, and aesthetics (or questions like 'which one is best or what makes it unique'), synthesize the visual details directly from '[Visual Scene, Composition & Details Analysis]'. Detail the subject pose, lighting, setting, color palette (e.g. pastel pink, champagne green, warm tones), materials, and artistic qualities objectively without fabricating unverified personas."
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
            clean_chunk_text = re.sub(r"\[Image URL:\s*.*?\]", "", chunk.text).strip()
            context_blocks.append(
                f"--- DOCUMENT EXCERPT {idx} ({chunk.source_file}) ---\n"
                f"{clean_chunk_text}"
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
