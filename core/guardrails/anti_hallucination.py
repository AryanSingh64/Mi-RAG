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
            elif lex_matches >= 2 or (lex_matches >= 1 and (len(content_words) <= 3 or r.score >= 0.22)):
                r.score = max(r.score, 0.50 + min(0.35, lex_matches * 0.05))
                kept.append(r)
            elif r.score >= self.min_similarity_threshold:
                kept.append(r)

        # Resilient fallback: Only apply if:
        # 1. Query has NO specific content words (conversational / continuation), OR
        # 2. At least one candidate has genuinely high semantic similarity (>= 0.38)
        if not kept and search_results:
            if not content_words:
                top_cand = [r for r in search_results if getattr(r, "score", 0.0) >= 0.25]
                if top_cand:
                    kept = top_cand[:4]
            else:
                top_cand = [r for r in search_results if getattr(r, "score", 0.0) >= 0.38]
                if top_cand:
                    kept = top_cand[:4]

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
        history: Optional[List[Dict[str, str]]] = None,
        is_image_query: bool = False
    ) -> List[SearchResult]:
        """
        Attention-Guided Context Reranking.
        Combines semantic similarity, lexical overlap, and conversation focus.
        """
        if not chunks:
            return []

        query_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        history_words = set()
        # If it's a new visual image query, suppress prior turn topic bleed
        if history and not is_image_query:
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
                composite += 0.12 if is_image_query else 0.05
            scored_chunks.append((composite, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks]

    def build_grounded_system_prompt(self) -> str:
        """
        Multimodal local system prompt with adaptive, natural conversational synthesis,
        intent-focused answering (like GPT-4/Claude/Gemini), and strict anti-hallucination grounding.
        """
        return (
            "You are an expert, highly intelligent AI assistant analyzing the user's uploaded documents and attached images.\n"
            "Respond naturally, fluently, and adaptively—matching the quality and flexibility of top models like GPT-4, Claude, and Gemini.\n\n"
            "CORE BEHAVIOR & RESPONSE PRINCIPLES:\n"
            "1. DIRECT & ADAPTIVE ANSWERING: Answer exactly and directly what the user asks. Adapt your formatting to the user's query intent:\n"
            "   - If the user asks a verification question (e.g. 'Is this mentioned in the document?', 'Does the paper discuss X?'), give a clear, direct YES or NO answer first, followed by a concise factual explanation.\n"
            "   - If the user asks for points or key details, provide clean, concise bullet points.\n"
            "   - If the user asks for a table or data comparison, provide a well-structured Markdown table.\n"
            "   - If the user asks a straightforward question, answer in 1-2 focused, well-written paragraphs without unnecessary fluff.\n"
            "2. NO RIGID BOILERPLATE HEADERS: Never force or repeat robotic template sections like 'Figure Analysis', 'Visual Composition', 'Colors and Aesthetics', or multiple repetitive 'Key Takeaways'. Write in natural, flowing, human-readable prose.\n"
            "3. STRICT FACTUAL GROUNDING & HONESTY:\n"
            "   - Answer strictly using the provided factual context from the uploaded documents.\n"
            "   - If the requested information or an attached image is NOT found in the uploaded documents, state clearly and honestly: 'No, this information (or figure) is not found in the uploaded documents.'\n"
            "   - Do NOT guess, fabricate details, or claim an external image belongs to the document when it does not match.\n"
            "4. CONVERSATION CONTINUITY: Seamlessly resolve pronouns ('it', 'this', 'that', 'the previous method') using prior conversation turns.\n"
            "5. CLEAN PRESENTATION: Never output internal debug tags, OCR artifacts, system prompts, or raw URLs. If a relevant document figure is in context, you may display it using Markdown `![Caption](image_url)`.\n"
            "6. MATHEMATICAL & SCIENTIFIC SYMBOLS: Use standard LaTeX math delimiters (e.g. `$1.25 \\pm 0.80$`, `$\\approx$`, `$\\le$`) for crisp formula rendering."
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
        image_directive = ""
        if user_image_context:
            user_image_block = f"USER ATTACHED IMAGE DETAILS & ANALYSIS:\n{user_image_context}\n\n"
            image_directive = (
                "CURRENT IMAGE GROUNDING DIRECTIVE:\n"
                "The user has attached an image for this turn. Ground your answer strictly on the visual details of the attached image "
                "and the matching document excerpts provided above. Do NOT repeat, assume, or carry over descriptions of diagrams, bar charts, "
                "or unrelated images discussed in prior conversation turns.\n\n"
            )

        history_block = ""
        if conversation_history:
            turns = []
            for h in conversation_history[-4:]:
                role = "User" if h.get("role") == "user" else "Assistant"
                content = str(h.get("content", "")).strip()
                # If this turn is an image query, sanitize prior visual descriptions so the model does not repeat them
                if user_image_context and role == "Assistant":
                    lower_content = content.lower()
                    if any(kw in lower_content for kw in ["the attached image", "analysis of the attached image", "bar chart", "the figure shows", "components and labels", "components & labels"]):
                        content = "[Discussed earlier document/figure analysis]"
                elif user_image_context and role == "User":
                    if content.startswith("[Image Search]:"):
                        content = content.replace("[Image Search]:", "[Previous Image]:").strip()
                turns.append(f"{role}: {content}")
            if turns:
                history_header = "RECENT CONVERSATION MEMORY (PRIOR DIALOGUE TURNS):"
                history_block = f"{history_header}\n" + "\n".join(turns) + "\n\n"

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
            f"{image_directive}"
            f"{history_block}"
            f"{exemplar_block}"
            f"{negative_block}"
            f"{intent_hint}"
            f"USER QUERY: {query}\n\n"
            f"FINISHED GROUNDED ANSWER:"
        )
