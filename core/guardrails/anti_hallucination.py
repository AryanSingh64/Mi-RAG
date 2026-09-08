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

    def filter_relevant_chunks(self, search_results: List[SearchResult], query: Optional[str] = None) -> List[SearchResult]:
        """
        Discards retrieved chunks that fall below the calibrated cosine similarity threshold,
        while strictly preserving and boosting chunks that contain exact lexical keyword matches.
        """
        if not search_results:
            return []

        content_words = set()
        if query:
            q_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
            stop_words = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "this", "that", "these", "those", "is", "are", "was", "were", "give", "tell", "show", "find", "the", "and", "for", "with", "from", "about", "table", "chart"}
            content_words = q_words - stop_words

        kept = []
        for r in search_results:
            chunk_lower = r.text.lower()
            has_exact = bool(query and len(query.strip()) >= 4 and query.strip().lower() in chunk_lower)
            lex_matches = sum(1 for w in content_words if w in chunk_lower) if content_words else 0
            
            if has_exact:
                r.score = max(r.score, 0.70)
                kept.append(r)
            elif lex_matches >= 1 and (len(content_words) <= 2 or lex_matches >= len(content_words) * 0.5):
                r.score = max(r.score, 0.50)
                kept.append(r)
            elif r.score >= self.min_similarity_threshold:
                kept.append(r)

        return kept

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
            r"\bno (?:information|reference|data|evidence|record|detail)\b",
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

        leading_text = ans_lower[:180].strip()
        leading_refusal_patterns = [
            r"^(?:i (?:could not|cannot) find|there is no (?:information|mention|record)|not (?:mentioned|found|present)|no direct evidence|i am sorry)",
            r"\bcould not find any information\b",
            r"\bcannot find any information\b",
            r"\bno direct evidence (?:was )?found\b",
            r"\bis not present in the (?:uploaded|provided) documents\b"
        ]
        has_leading_refusal = any(re.search(pat, leading_text) for pat in leading_refusal_patterns)

        # Only flag as negative response if the response is fundamentally a refusal
        is_negative_response = False
        if has_leading_refusal:
            is_negative_response = True
        elif len(answer.strip()) < 220 and any(re.search(pat, ans_lower) for pat in negative_patterns):
            is_negative_response = True
        
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
        if calibrated_score < 0.20 and not has_lexical and len(content_query_words) > 0:
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
            overlap = sum(1 for w in query_words if w in chunk_lower)
            history_overlap = sum(1 for w in history_words if w in chunk_lower)
            has_fig = bool(chunk.metadata.get("has_image") or chunk.metadata.get("image_url") if isinstance(chunk.metadata, dict) else False)

            # Composite attention score
            composite = chunk.score * 0.65 + min(overlap * 0.08, 0.25) + min(history_overlap * 0.04, 0.10)
            if has_fig:
                composite += 0.05
            scored_chunks.append((composite, chunk))

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
            "8. CONTEXTUAL INLINE DIAGRAMS & FIGURES: When discussing or explaining an attached figure, diagram, or chart from the context, embed it directly within your explanation at the relevant point using Markdown: ![Figure Caption](image_url). Place the figure immediately below the paragraph or heading that explains it so the visual proof is in context.\n"
            "9. ANTI-FABRICATION: Ground all explanations strictly in the factual excerpts provided. Do not fabricate unmentioned facts or entities. Answer directly without repeating system directives or meta-commentary about what is unmentioned.\n"
            "10. VISUAL COMPOSITION, COLORS & LOGICAL AESTHETICS: When the user asks what an image represents, what is special in it, or asks about composition, colors, and aesthetics, synthesize the visual details directly from '[Visual Scene, Composition & Details Analysis]'.\n"
            "11. DIRECTNESS & INTENT: Answer specifically what the user asks. If the user asks about a specific diagram, table, formula, or concept (e.g. 'model training diagram'), focus strictly on explaining that exact item. Do NOT reproduce a boilerplate paper template (Abstract, Introduction, Methods, Results, Conclusion) unless the user explicitly asks for a general paper overview."
        )

    def build_user_prompt(
        self,
        query: str,
        context_chunks: List[SearchResult],
        user_image_context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        feedback_exemplars: Optional[List[Dict[str, str]]] = None,
        negative_exemplars: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Formats retrieved chunks, conversation memory, feedback exemplars, negative constraints,
        and optional attached query image into clear context for the LLM.
        """
        context_blocks = []
        for idx, chunk in enumerate(context_chunks, start=1):
            clean_chunk_text = chunk.text
            has_fig = False
            img_url = ""
            img_caption = ""

            # Detect image URL from metadata or text
            if isinstance(chunk.metadata, dict):
                img_url = chunk.metadata.get("image_url", "")
                img_caption = chunk.metadata.get("caption", "") or chunk.metadata.get("title", "")
                if img_url or chunk.metadata.get("has_image"):
                    has_fig = True

            url_match = re.search(r"\[Image URL:\s*(.*?)\]", chunk.text)
            if url_match:
                img_url = img_url or url_match.group(1).strip()
                has_fig = True

            cap_match = re.search(r"\[IMAGE / FIGURE:\s*(.*?)\]", chunk.text, re.IGNORECASE)
            if cap_match:
                img_caption = img_caption or cap_match.group(1).strip()
                has_fig = True

            clean_chunk_text = re.sub(r"\[Image URL:\s*.*?\]", "", clean_chunk_text).strip()
            
            fig_instruction = ""
            if has_fig and img_url:
                cap_label = img_caption or f"Figure from {chunk.source_file}"
                fig_instruction = f"\n[ATTACHED DIAGRAM AVAILABLE - Embed inline where discussed using: ![{cap_label}]({img_url})]"

            fig_tag = " [Attached Figure Available]" if has_fig else ""
            context_blocks.append(
                f"--- DOCUMENT EXCERPT {idx} ({chunk.source_file}{fig_tag}) ---\n"
                f"{clean_chunk_text}{fig_instruction}"
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

        negative_block = ""
        if negative_exemplars:
            neg_items = []
            for neg in negative_exemplars[:1]:
                q = neg.get("query", "").strip()
                a = neg.get("answer", "").strip()
                if q and a:
                    short_a = a[:220].replace("\n", " ")
                    neg_items.append(
                        f"Previous Query: {q}\n"
                        f"Rejected Flawed Output: \"{short_a}...\"\n"
                        f"Correction Directive: Do NOT repeat the flaws, vague statements, or inaccuracies from the rejected output above. "
                        f"Provide a comprehensive, factually precise response directly grounded in the document excerpts."
                    )
            if neg_items:
                negative_block = "PREVIOUSLY REJECTED ANSWER PATTERNS (NEGATIVE CONSTRAINTS TO AVOID):\n" + "\n\n".join(neg_items) + "\n\n"

        intent_hint = ""
        q_lower = query.lower().strip()
        is_general_summary = any(k in q_lower for k in ["summary", "summarise", "overview", "give me a summary", "what is this paper about"])
        if not is_general_summary and any(k in q_lower for k in ["diagram", "figure", "image", "chart", "table", "graph", "histogram", "architecture", "training", "model", "pipeline", "what is", "how does", "why"]):
            intent_hint = (
                f"TARGETED FOCUS DIRECTIVE: The user is asking specifically about: '{query}'. "
                f"Directly explain and address this specific item. Do NOT output a generic paper overview "
                f"(do NOT generate Abstract, Introduction, Methods, Results, Conclusion sections) unless specifically asked for a full paper summary.\n\n"
            )

        return (
            f"KNOWLEDGE BASE CONTEXT:\n"
            f"{formatted_context}\n\n"
            f"{user_image_block}"
            f"{history_block}"
            f"{exemplar_block}"
            f"{negative_block}"
            f"{intent_hint}"
            f"USER QUERY: {query}\n\n"
            f"FINISHED GROUNDED ANSWER:"
        )
