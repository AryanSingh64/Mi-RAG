import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from core.chunking.text_chunker import RecursiveChunker
from core.embeddings.embedder import LocalEmbedder
from core.guardrails.anti_hallucination import AntiHallucinationEngine, GroundedAnswer
from core.guardrails.query_rewriter import QueryRewriter
from core.ingestion.factory import DocumentParserFactory
from core.llm.ollama_client import OllamaClient
from core.llm.multi_provider import MultiProviderLLM
from core.vectorstore.chroma_store import ChromaVectorStore, SearchResult
from core.vectorstore.image_matcher import VisualFingerprintMatcher


class RAGPipeline:
    """
    Unified end-to-end RAG orchestrator with Query Auto-Correction & Anti-Hallucination.
    """

    def __init__(
        self,
        persist_directory: Path | str,
        collection_name: str = "knowledge_base",
        embedding_model: str = "all-MiniLM-L6-v2",
        ollama_model: str = "llama3.2:3b",
        vision_models: Optional[List[str] | str] = None,
        ollama_url: str = "http://localhost:11434",
        extracted_images_dir: Optional[Path | str] = None,
        session_id: Optional[str] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 60,
        min_similarity_threshold: Optional[float] = None,
        strictness: str = "balanced",
        missing_answer_behavior: str = "refusal",
        response_style: str = "detailed"
    ):
        self.session_id = session_id
        self.strictness = (strictness or "balanced").lower().strip()
        self.missing_answer_behavior = (missing_answer_behavior or "refusal").lower().strip()
        self.response_style = (response_style or "detailed").lower().strip()

        if min_similarity_threshold is not None:
            effective_threshold = min_similarity_threshold
        elif self.strictness == "strict":
            effective_threshold = 0.42
        elif self.strictness == "creative":
            effective_threshold = 0.22
        else:  # balanced
            effective_threshold = 0.35

        self.min_similarity_threshold = effective_threshold
        self.extracted_images_dir = Path(extracted_images_dir) if extracted_images_dir else None
        self.vision_models = vision_models or ["moondream"]
        self.parser_factory = DocumentParserFactory(
            vision_models=self.vision_models,
            output_images_dir=self.extracted_images_dir,
            session_id=self.session_id
        )
        self.chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_model = embedding_model
        self.embedder = LocalEmbedder(model_name=embedding_model)
        self.vector_store = ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedder=self.embedder
        )
        self.ollama = OllamaClient(base_url=ollama_url, default_model=ollama_model)
        self.guardrails = AntiHallucinationEngine(min_similarity_threshold=self.min_similarity_threshold)
        self.rewriter = QueryRewriter(ollama_client=self.ollama)
        self.current_model = ollama_model
        self.persist_directory = Path(persist_directory)
        self.feedback_file = self.persist_directory.parent / "feedback.json"
        self.global_feedback_file = Path.home() / ".mirag" / "global_feedback.json"
        self.citation_boosts: Dict[str, float] = {}
        self.conversation_memory: List[Dict[str, str]] = []
        self.feedback_store: List[Dict[str, Any]] = self._load_feedback()

    def _load_feedback(self) -> List[Dict[str, Any]]:
        """Loads persistent feedback from session directory or global store."""
        import json
        loaded = []
        if self.feedback_file.exists():
            try:
                loaded = json.loads(self.feedback_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        if not loaded and self.global_feedback_file.exists():
            try:
                loaded = json.loads(self.global_feedback_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        # Populate initial citation boosts from loaded feedback
        for entry in loaded:
            r = entry.get("rating")
            cits = entry.get("citations", [])
            delta = 0.12 if r == "thumbs_up" else -0.15
            for c in cits:
                self.citation_boosts[str(c)] = round(self.citation_boosts.get(str(c), 0.0) + delta, 3)

        return loaded

    def record_feedback(
        self,
        query: str,
        answer: str,
        rating: str,
        model: Optional[str] = None,
        citations: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Records user feedback (thumbs_up / thumbs_down) to actively improve RAG accuracy:
        1. Thumbs Up: Rewards source citations and saves Q&A as an in-context few-shot exemplar.
        2. Thumbs Down: Penalizes citations, broadens future retrieval, and injects negative constraints.
        3. Persists feedback locally and globally for DPO / SFT dataset export.
        """
        import time
        import json
        clean_cits = []
        if citations:
            for c in citations:
                if isinstance(c, dict):
                    clean_cits.append(c.get("source_file", str(c)))
                elif hasattr(c, "source_file"):
                    clean_cits.append(c.source_file)
                else:
                    clean_cits.append(str(c))

        entry = {
            "query": query.strip(),
            "answer": answer.strip(),
            "rating": rating,
            "model": model or self.current_model,
            "citations": clean_cits,
            "timestamp": time.time()
        }
        self.feedback_store.append(entry)

        # Active citation reinforcement / penalty
        delta = 0.12 if rating == "thumbs_up" else -0.15
        for c in clean_cits:
            self.citation_boosts[str(c)] = round(self.citation_boosts.get(str(c), 0.0) + delta, 3)

        # Persist feedback locally to session and to global ~/.mirag dataset
        try:
            self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
            self.feedback_file.write_text(json.dumps(self.feedback_store, indent=2), encoding="utf-8")
            self.global_feedback_file.parent.mkdir(parents=True, exist_ok=True)
            self.global_feedback_file.write_text(json.dumps(self.feedback_store, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[!] Feedback persistence notice: {e}")

        print(f"[*] FEEDBACK RECORDED [{rating.upper()}]: Query '{query[:45]}...' -> {len(self.feedback_store)} feedback items in store (Citations updated)")
        return entry

    def get_feedback_exemplars(self, current_query: str, max_exemplars: int = 2) -> List[Dict[str, str]]:
        """Retrieves top verified thumbs_up exemplars ranked by relevance for dynamic few-shot in-context learning."""
        approved = [f for f in self.feedback_store if f.get("rating") == "thumbs_up"]
        if not approved:
            return []

        q_words = set(re.findall(r"\b\w{3,}\b", current_query.lower()))
        stop_words = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "this", "that", "these", "those", "is", "are", "was", "were", "give", "tell", "show", "find", "the", "and", "for", "with", "from", "about"}
        content_words = q_words - stop_words
        if not content_words:
            return []

        # Rank approved items by keyword overlap with current query
        scored = []
        for item in approved:
            item_q = item.get("query", "").lower()
            overlap = sum(1 for w in content_words if w in item_q)
            # Strictly require overlap >= 1 to prevent unrelated cross-document topic contamination
            if overlap >= 1:
                scored.append((overlap, item))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        top_items = [item for _, item in scored[:max_exemplars]]
        return [{"query": item["query"], "answer": item["answer"]} for item in top_items]

    def get_negative_exemplars(self, current_query: str, max_exemplars: int = 1) -> List[Dict[str, str]]:
        """Retrieves rejected thumbs_down exemplars for queries similar to the current one to inject negative constraints."""
        rejected = [f for f in self.feedback_store if f.get("rating") == "thumbs_down"]
        if not rejected:
            return []

        q_words = set(re.findall(r"\b\w{3,}\b", current_query.lower()))
        if not q_words:
            return []

        # Only inject if the query has meaningful lexical overlap with the rejected question
        scored = []
        for item in rejected:
            item_q = item.get("query", "").lower()
            overlap = sum(1 for w in q_words if w in item_q)
            if overlap >= 1:
                scored.append((overlap, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"query": item["query"], "answer": item["answer"]} for _, item in scored[:max_exemplars]]

    def clear_memory(self) -> None:
        """Resets the multi-turn conversational dialogue memory."""
        self.conversation_memory = []
        print("[*] Conversational memory cleared.")

    def ingest_file(
        self,
        file_path: Path | str,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        progress_callback: Optional[Any] = None
    ) -> int:
        """
        Parses a file (text, docx, PDF with images/OCR, standalone images),
        chunks it, and indexes it into ChromaDB.
        """
        fname = Path(file_path).name
        print(f"[*] Step 1/3: Parsing document & extracting images: {fname}...", flush=True)
        parsed_doc = self.parser_factory.parse_file(
            file_path,
            start_page=start_page,
            end_page=end_page,
            progress_callback=progress_callback
        )
        
        page_info = f"({parsed_doc.metadata.get('total_pages', 1)} pages analyzed"
        if "total_doc_pages" in parsed_doc.metadata:
            page_info += f" of {parsed_doc.metadata['total_doc_pages']}"
        page_info += f", {len(parsed_doc.text_content):,} characters)"

        print(f"[*] Step 2/3: Chunking document into semantic passages {page_info}...", flush=True)
        if progress_callback:
            progress_callback("chunking", 1, 1, parsed_doc.metadata.get("diagram_count", 0))
        chunks = self.chunker.chunk_document(parsed_doc)
        
        print(f"[*] Step 3/3: Storing {len(chunks)} chunk vectors in ChromaDB with {self.embedding_model}...", flush=True)
        self.vector_store.add_chunks(chunks, progress_callback=progress_callback)
        return len(chunks)

    def query(
        self,
        user_question: str,
        top_k: int = 6,
        history: Optional[List[Dict[str, str]]] = None,
        provider: str = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> GroundedAnswer:
        """
        Executes a query with spelling auto-correction, attention-guided reranking,
        multi-turn conversation memory, and anti-hallucination.
        """
        if history is not None:
            self.conversation_memory = history[-10:]

        target_model = model or self.current_model
        provider_name = (provider or "ollama").lower().strip()

        # 1. Handle casual greetings ("hi", "hello") politely
        greeting_reply = self.rewriter.is_conversational_greeting(user_question)
        if greeting_reply:
            self.conversation_memory.append({"role": "user", "content": user_question})
            self.conversation_memory.append({"role": "assistant", "content": greeting_reply})
            return GroundedAnswer(
                answer=greeting_reply,
                is_grounded=True,
                confidence_score=1.0,
                citations=[]
            )

        # 2. Fix typos & expand query with conversation attention
        cleaned_question, search_query = self.rewriter.clean_and_expand_query(
            user_question,
            model_name=target_model
        )

        q_lower = cleaned_question.lower()
        is_summary_query = any(w in q_lower for w in ["summary", "summarize", "overview", "what is this document", "what is this paper", "explain this document"])
        is_visual_query = any(w in q_lower for w in ["image", "images", "figure", "figures", "diagram", "diagrams", "plot", "plots", "chart", "charts", "picture", "pictures", "visual"])

        # Check if negative feedback exists for this query; if so, broaden retrieval top_k
        negative_exemplars = self.get_negative_exemplars(cleaned_question)
        effective_top_k = top_k + 2 if negative_exemplars else top_k

        # 3. Multi-strategy retrieval (corrected query + original query + keyword search)
        raw_results = self.vector_store.query(search_query, top_k=effective_top_k)
        if search_query != user_question:
            direct_results = self.vector_store.query(user_question, top_k=effective_top_k)
            seen = {r.chunk_id: r for r in raw_results}
            for dr in direct_results:
                if dr.chunk_id not in seen:
                    raw_results.append(dr)

        # Keyword search augmentation for exact technical terms (e.g. "confusion matrix", "keywords", "oncotype")
        if hasattr(self.vector_store, "keyword_search"):
            kw_results = self.vector_store.keyword_search(cleaned_question, top_k=3)
            seen = {r.chunk_id: r for r in raw_results}
            for kr in kw_results:
                if kr.chunk_id not in seen:
                    raw_results.append(kr)

        # Broad overview / summary query augmentation: pull in lead document chunks (abstract, intro)
        if is_summary_query and hasattr(self.vector_store, "get_initial_chunks"):
            initial_chunks = self.vector_store.get_initial_chunks(limit=3)
            seen = {r.chunk_id: r for r in raw_results}
            for ic in initial_chunks:
                if ic.chunk_id not in seen:
                    raw_results.append(ic)

        # Visual query augmentation: pull in chunks containing figures/diagrams
        if is_visual_query and hasattr(self.vector_store, "get_image_chunks"):
            fig_chunks = self.vector_store.get_image_chunks(limit=6)
            seen = {r.chunk_id: r for r in raw_results}
            for fc in fig_chunks:
                if fc.chunk_id not in seen:
                    raw_results.append(fc)

        # 4. Filter with lexical keyword preservation and apply Attention-Guided Context Reranking
        filtered_chunks = self.guardrails.filter_relevant_chunks(raw_results, query=cleaned_question)

        # Active feedback-guided citation weighting (boosts thumbs_up sources, demotes thumbs_down sources)
        if self.citation_boosts and filtered_chunks:
            for c in filtered_chunks:
                boost = self.citation_boosts.get(c.source_file, 0.0)
                if boost:
                    c.score = max(0.0, min(1.0, c.score + boost))

        relevant_chunks = self.guardrails.apply_attention_reranking(
            query=cleaned_question,
            chunks=filtered_chunks,
            history=self.conversation_memory
        )

        print("\n" + "="*60)
        print(f"[*] RAG RETRIEVAL ENGINE: '{cleaned_question}' (Provider: {provider_name.upper()} | Model: {target_model})")
        if relevant_chunks:
            for idx, chunk in enumerate(relevant_chunks, 1):
                preview = chunk.text.replace("\n", " ")[:90]
                print(f"  [{idx}] {chunk.source_file} (Similarity: {chunk.score*100:.1f}%) -> {preview}...")
        else:
            print("[!] No relevant chunks met the similarity threshold.")
        print("="*60 + "\n")

        if not relevant_chunks:
            if self.missing_answer_behavior == "general_knowledge":
                gen_prompt = (
                    f"Answer the following user question using your general knowledge: {cleaned_question}\n\n"
                    "Requirement: You MUST start your response with: '[General Knowledge - Not found in your documents]'."
                )
                try:
                    llm_resp = MultiProviderLLM.generate(
                        user_prompt=gen_prompt,
                        system_prompt="You are a helpful AI assistant answering from general knowledge because the topic was not found in the user's private documents.",
                        provider=provider_name,
                        model=target_model,
                        api_key=api_key,
                        temperature=0.3,
                        ollama_url=self.ollama.base_url
                    )
                except Exception:
                    llm_resp = f"I could not find any information about '{user_question}' in the uploaded documents."
                self.conversation_memory.append({"role": "user", "content": user_question})
                self.conversation_memory.append({"role": "assistant", "content": llm_resp})
                return GroundedAnswer(
                    answer=llm_resp,
                    is_grounded=False,
                    confidence_score=0.25,
                    citations=[],
                    warning="Answered from general knowledge."
                )
            else:
                fallback_msg = f"I could not find any information about '{user_question}' in the uploaded documents. Please upload a document containing this topic or ask about your indexed files."
                self.conversation_memory.append({"role": "user", "content": user_question})
                self.conversation_memory.append({"role": "assistant", "content": fallback_msg})
                return GroundedAnswer(
                    answer=fallback_msg,
                    is_grounded=False,
                    confidence_score=0.0,
                    citations=[],
                    warning="No relevant chunks met the threshold."
                )

        # 5. Construct grounded prompt with conversation memory, positive exemplars & negative constraints
        system_prompt = self.guardrails.build_grounded_system_prompt()
        if self.response_style == "concise":
            system_prompt += "\n\nRESPONSE FORMAT DIRECTIVE: Deliver a concise, direct answer using bullet points. Avoid filler introductions or long summaries."
        elif self.response_style == "detailed":
            system_prompt += "\n\nRESPONSE FORMAT DIRECTIVE: Deliver a comprehensive, in-depth explanation with complete context and clear headings."
        feedback_exemplars = self.get_feedback_exemplars(cleaned_question)
        user_prompt = self.guardrails.build_user_prompt(
            query=cleaned_question,
            context_chunks=relevant_chunks,
            conversation_history=self.conversation_memory,
            feedback_exemplars=feedback_exemplars,
            negative_exemplars=negative_exemplars
        )

        # 6. Query LLM via MultiProvider dispatcher
        try:
            llm_response = MultiProviderLLM.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                provider=provider_name,
                model=target_model,
                api_key=api_key,
                temperature=0.1,
                ollama_url=self.ollama.base_url
            )
            llm_response = self._sanitize_llm_response(llm_response)
        except Exception as e:
            return GroundedAnswer(
                answer=f"Error communicating with {provider_name.upper()}: {str(e)}",
                is_grounded=False,
                confidence_score=0.0,
                citations=relevant_chunks,
                warning=f"{provider_name.upper()} connection note: {str(e)}"
            )

        # Record conversation turn in memory
        self.conversation_memory.append({"role": "user", "content": user_question})
        self.conversation_memory.append({"role": "assistant", "content": llm_response})
        if len(self.conversation_memory) > 12:
            self.conversation_memory = self.conversation_memory[-12:]

        # Evaluate genuine factual grounding
        is_grounded, confidence_score, grounded_citations = self.guardrails.evaluate_grounding(
            query=cleaned_question,
            answer=llm_response,
            relevant_chunks=relevant_chunks
        )

        # Extract visually relevant diagrams
        matched_images = []
        if is_grounded or is_visual_query:
            matched_images = self._extract_relevant_images(user_question, relevant_chunks, is_image_query=is_visual_query)

        # Fallback for visual query if relevant_chunks yielded no images
        if is_visual_query and not matched_images and hasattr(self.vector_store, "get_image_chunks"):
            img_pool = self.vector_store.get_image_chunks(limit=6)
            if img_pool:
                matched_images = self._extract_relevant_images(user_question, img_pool, is_image_query=True)

        if is_visual_query and matched_images and not is_grounded:
            is_grounded = True
            confidence_score = max(0.65, confidence_score)

        if matched_images:
            print(f"[*] ATTACHED VISUAL DIAGRAMS ({len(matched_images)}):")
            for img in matched_images:
                print(f"    - {img['source_file']} -> {img['url']}")

        return GroundedAnswer(
            answer=llm_response,
            is_grounded=is_grounded,
            confidence_score=confidence_score,
            citations=grounded_citations if grounded_citations else relevant_chunks[:3],
            images=matched_images
        )

    def _find_chunks_for_image_match(self, query_image_path: Path, min_threshold: float = 0.85) -> Tuple[List[SearchResult], List[str], Optional[Dict[str, Any]]]:
        """
        Performs high-speed perceptual visual matching against extracted document figures.
        If a match is found, retrieves corresponding document chunks and figure metadata.
        Returns: (matched_chunks, info_lines, top_matched_image_dict)
        """
        matched_chunks: List[SearchResult] = []
        info_lines: List[str] = []
        top_image_match: Optional[Dict[str, Any]] = None

        if not self.extracted_images_dir:
            return matched_chunks, info_lines, top_image_match

        ext_dir = Path(self.extracted_images_dir)
        if not ext_dir.exists():
            return matched_chunks, info_lines, top_image_match

        visual_matches = VisualFingerprintMatcher.match_against_directory(
            query_image=query_image_path,
            target_dir=ext_dir,
            min_threshold=min_threshold,
            top_k=2
        )
        if not visual_matches:
            return matched_chunks, info_lines, top_image_match

        top_fig_path, top_sim = visual_matches[0]
        fig_filename = top_fig_path.name
        print(f"[*] Perceptual visual match: {fig_filename} (similarity: {top_sim*100:.1f}%)")

        img_url = f"/api/sessions/{self.session_id}/images/{fig_filename}" if self.session_id else f"/images/{fig_filename}"
        top_image_match = {
            "url": img_url,
            "filename": fig_filename,
            "source_file": fig_filename,
            "relevance": round(min(100.0, top_sim * 100.0), 1),
            "overlap": 10
        }

        if not hasattr(self.vector_store, "collection") or not self.vector_store.collection:
            return matched_chunks, info_lines, top_image_match

        try:
            all_data = self.vector_store.collection.get(include=["documents", "metadatas"])
            matched_pages = set()
            source_file = ""

            # Pass 1: Chunks directly linking or naming this figure
            primary_caption = ""
            for doc_text, meta in zip(all_data["documents"], all_data["metadatas"]):
                meta_url = meta.get("image_url", "") if isinstance(meta, dict) else ""
                if fig_filename in meta_url or fig_filename in doc_text:
                    p_num = meta.get("page_number", "") if isinstance(meta, dict) else ""
                    s_file = meta.get("source_file", "") if isinstance(meta, dict) else ""
                    if p_num:
                        matched_pages.add(p_num)
                    if s_file:
                        source_file = s_file

                    cap_match = re.search(r"\[IMAGE / FIGURE:\s*(.*?)\]", doc_text, re.IGNORECASE)
                    cap = cap_match.group(1).strip() if cap_match else (meta.get("caption", "") if isinstance(meta, dict) else "")
                    if cap and not primary_caption and not cap.startswith("Figure (Page Image"):
                        primary_caption = cap

                    score = max(0.95, top_sim)
                    matched_chunks.append(SearchResult(
                        chunk_id=f"fig_exact_{fig_filename}_{meta.get('chunk_index', 0)}",
                        text=doc_text,
                        source_file=s_file or fig_filename,
                        metadata=meta if isinstance(meta, dict) else {},
                        score=score,
                        distance=1.0 - score
                    ))

            # Pass 2: Adjacent context on the same page explaining this figure
            if matched_pages and source_file:
                for doc_text, meta in zip(all_data["documents"], all_data["metadatas"]):
                    if not isinstance(meta, dict):
                        continue
                    if meta.get("source_file") == source_file and meta.get("page_number") in matched_pages:
                        if not primary_caption:
                            fig_text_match = re.search(r"(?:Fig(?:ure)?\.?\s*\d+[^.\n]+(?:\.[^.\n]+)?)", doc_text)
                            if fig_text_match:
                                primary_caption = fig_text_match.group(0).strip()

                        c_id = f"fig_page_{fig_filename}_{meta.get('chunk_index', 0)}"
                        if any(mc.text == doc_text for mc in matched_chunks):
                            continue
                        score = max(0.85, top_sim - 0.10)
                        matched_chunks.append(SearchResult(
                            chunk_id=c_id,
                            text=doc_text,
                            source_file=source_file,
                            metadata=meta,
                            score=score,
                            distance=1.0 - score
                        ))

            first_p_num = sorted(matched_pages)[0] if matched_pages else ""
            label = f"'{primary_caption}'" if primary_caption else f"Figure '{fig_filename}'"
            info_lines.append(f"Visual match found in document: {label} ({source_file or 'document'}, Page {first_p_num}) with {top_sim*100:.1f}% confidence.")
            top_image_match["source_file"] = f"{source_file} (Page {first_p_num})" if (source_file and first_p_num) else (source_file or fig_filename)
        except Exception as e:
            print(f"[*] Note during visual chunk lookup: {e}")

        return matched_chunks, info_lines, top_image_match

    def query_with_image(
        self,
        user_question: str,
        query_image_path: Path | str,
        top_k: int = 6,
        history: Optional[List[Dict[str, str]]] = None,
        provider: str = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> GroundedAnswer:
        """
        Executes a Multimodal Visual Search & Query with memory, perceptual image matching, and attention.
        """
        if history is not None:
            self.conversation_memory = history[-10:]

        target_model = model or self.current_model
        provider_name = (provider or "ollama").lower().strip()

        image_path = Path(query_image_path)
        print(f"\n[*] MULTIMODAL QUERY WITH ATTACHED IMAGE: {image_path.name} (Provider: {provider_name.upper()})")

        # 1. Perceptual visual fingerprint matching against document extracted figures
        fig_chunks, fig_info, top_image_match = self._find_chunks_for_image_match(image_path)

        # 2. Extract OCR text and Vision description of query image
        vision_parser = getattr(self.parser_factory, "vision_parser", getattr(self.parser_factory, "_image_parser", None))
        if vision_parser:
            image_analysis = vision_parser.describe_and_ocr_image(image_path)
        else:
            image_analysis = {"ocr_text": "", "description": "", "combined_summary": ""}

        ocr_text = image_analysis.get("ocr_text", "")
        vision_desc = image_analysis.get("description", "")
        combined_summary = image_analysis.get("combined_summary", "")

        if fig_info:
            vis_match_block = "\n".join(fig_info)
            combined_summary = f"{vis_match_block}\n\n{combined_summary}" if combined_summary else vis_match_block
        else:
            unmatched_note = (
                "[ATTACHED IMAGE STATUS]: This attached image does NOT match any figure or diagram in the uploaded document(s). "
                "It is an external or distinct image provided by the user."
            )
            combined_summary = f"{unmatched_note}\n\n{combined_summary}" if combined_summary else unmatched_note

        # 3. Formulate enriched search query
        effective_question = user_question.strip() if user_question else "What is this image and how does it relate to the uploaded documents?"
        
        # Formulate visual search topic from visual match, OCR text & vision model descriptions
        visual_terms = []
        if fig_info:
            visual_terms.append(" ".join(fig_info))
        if ocr_text:
            clean_ocr = " ".join([w for w in ocr_text.split() if len(w) > 1 or w.isalnum()])
            if clean_ocr:
                visual_terms.append(clean_ocr)
        if vision_desc:
            clean_desc = vision_desc[:300].strip()
            if clean_desc:
                visual_terms.append(clean_desc)

        visual_search_topic = f"{effective_question} {' '.join(visual_terms)}".strip()

        # 4. Retrieve relevant chunks from ChromaDB using visual_search_topic
        retrieval_k = max(top_k * 2, 12)
        raw_results = self.vector_store.query(visual_search_topic, top_k=retrieval_k)

        # Prepend visually matched figure chunks so they are prioritized
        seen_texts = {r.text for r in raw_results}
        for fc in reversed(fig_chunks):
            if fc.text not in seen_texts:
                raw_results.insert(0, fc)
                seen_texts.add(fc.text)

        filtered_chunks = self.guardrails.filter_relevant_chunks(raw_results, query=visual_search_topic)
        # Ensure visually matched figure chunks remain in filtered_chunks
        filt_texts = {c.text for c in filtered_chunks}
        for fc in reversed(fig_chunks):
            if fc.text not in filt_texts:
                filtered_chunks.insert(0, fc)
                filt_texts.add(fc.text)

        relevant_chunks = self.guardrails.apply_attention_reranking(
            query=visual_search_topic,
            chunks=filtered_chunks,
            history=self.conversation_memory,
            is_image_query=True
        )[:top_k]

        # 5. Construct prompt with user image analysis & conversation memory & feedback exemplars
        system_prompt = self.guardrails.build_grounded_system_prompt()
        feedback_exemplars = self.get_feedback_exemplars(effective_question)
        user_prompt = self.guardrails.build_user_prompt(
            query=effective_question,
            context_chunks=relevant_chunks,
            user_image_context=combined_summary,
            conversation_history=self.conversation_memory,
            feedback_exemplars=feedback_exemplars
        )

        # 6. Query LLM via MultiProvider dispatcher
        try:
            llm_response = MultiProviderLLM.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                provider=provider_name,
                model=target_model,
                api_key=api_key,
                temperature=0.1,
                ollama_url=self.ollama.base_url
            )
            llm_response = self._sanitize_llm_response(llm_response)
        except Exception as e:
            return GroundedAnswer(
                answer=f"Error communicating with {provider_name.upper()}: {str(e)}",
                is_grounded=False,
                confidence_score=0.0,
                citations=relevant_chunks,
                warning=f"{provider_name.upper()} connection note: {str(e)}"
            )

        # Record conversation turn in memory
        self.conversation_memory.append({"role": "user", "content": f"[Image Search]: {effective_question}"})
        self.conversation_memory.append({"role": "assistant", "content": llm_response})
        if len(self.conversation_memory) > 12:
            self.conversation_memory = self.conversation_memory[-12:]

        # Evaluate genuine factual grounding against visual search topic
        is_grounded, confidence_score, grounded_citations = self.guardrails.evaluate_grounding(
            query=visual_search_topic,
            answer=llm_response,
            relevant_chunks=relevant_chunks
        )

        # Ensure reverse image queries with strong visual match or relevant context are grounded
        if (top_image_match or fig_chunks) and not is_grounded and relevant_chunks:
            is_grounded = True
            confidence_score = max(confidence_score, 0.78)
            grounded_citations = relevant_chunks
        elif not is_grounded and relevant_chunks:
            top_score = max(getattr(c, "score", 0.0) for c in relevant_chunks)
            if top_score >= 0.40:
                is_grounded = True
                confidence_score = max(confidence_score, 0.65)
                grounded_citations = relevant_chunks

        # 7. Extract top matching document diagrams (capped at 2-3 most relevant)
        matched_images = []
        if top_image_match:
            matched_images.append(top_image_match)
        if is_grounded or relevant_chunks:
            extra_imgs = self._extract_relevant_images(visual_search_topic, relevant_chunks, is_image_query=True)
            for ei in extra_imgs:
                if not any(m["url"] == ei["url"] for m in matched_images):
                    matched_images.append(ei)
            matched_images = matched_images[:3]
            if matched_images and not is_grounded:
                is_grounded = True
                confidence_score = max(confidence_score, 0.65)
                grounded_citations = relevant_chunks

        return GroundedAnswer(
            answer=llm_response,
            is_grounded=is_grounded,
            confidence_score=confidence_score,
            citations=grounded_citations,
            images=matched_images
        )

    def query_stream(
        self,
        user_question: str,
        top_k: int = 6,
        history: Optional[List[Dict[str, str]]] = None,
        provider: str = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Executes a real-time token-by-token streaming RAG query.
        Yields {"type": "token", "token": "..."} during generation,
        followed by {"type": "done", ...} with final grounded metadata.
        """
        if history is not None:
            self.conversation_memory = history[-10:]

        target_model = model or self.current_model
        provider_name = (provider or "ollama").lower().strip()

        # 1. Handle casual greetings politely
        greeting_reply = self.rewriter.is_conversational_greeting(user_question)
        if greeting_reply:
            self.conversation_memory.append({"role": "user", "content": user_question})
            self.conversation_memory.append({"role": "assistant", "content": greeting_reply})
            words = greeting_reply.split(" ")
            for i, w in enumerate(words):
                yield {"type": "token", "token": w + (" " if i < len(words) - 1 else "")}
            yield {
                "type": "done",
                "answer": greeting_reply,
                "confidence_score": 1.0,
                "is_grounded": True,
                "citations": [],
                "images": []
            }
            return

        # 2. Fix typos & expand query with conversation attention
        cleaned_question, search_query = self.rewriter.clean_and_expand_query(
            user_question,
            model_name=target_model
        )

        q_lower = cleaned_question.lower()
        is_summary_query = any(w in q_lower for w in ["summary", "summarize", "overview", "what is this document", "what is this paper", "explain this document"])
        is_visual_query = any(w in q_lower for w in ["image", "images", "figure", "figures", "diagram", "diagrams", "plot", "plots", "chart", "charts", "picture", "pictures", "visual"])

        # 3. Multi-strategy retrieval (corrected query + original query + keyword search)
        negative_exemplars = self.get_negative_exemplars(cleaned_question)
        effective_top_k = top_k + 2 if negative_exemplars else top_k

        raw_results = self.vector_store.query(search_query, top_k=effective_top_k)
        if search_query != user_question:
            direct_results = self.vector_store.query(user_question, top_k=effective_top_k)
            seen = {r.chunk_id: r for r in raw_results}
            for dr in direct_results:
                if dr.chunk_id not in seen:
                    raw_results.append(dr)

        if hasattr(self.vector_store, "keyword_search"):
            kw_results = self.vector_store.keyword_search(cleaned_question, top_k=3)
            seen = {r.chunk_id: r for r in raw_results}
            for kr in kw_results:
                if kr.chunk_id not in seen:
                    raw_results.append(kr)

        if is_summary_query and hasattr(self.vector_store, "get_initial_chunks"):
            initial_chunks = self.vector_store.get_initial_chunks(limit=3)
            seen = {r.chunk_id: r for r in raw_results}
            for ic in initial_chunks:
                if ic.chunk_id not in seen:
                    raw_results.append(ic)

        if is_visual_query and hasattr(self.vector_store, "get_image_chunks"):
            fig_chunks = self.vector_store.get_image_chunks(limit=6)
            seen = {r.chunk_id: r for r in raw_results}
            for fc in fig_chunks:
                if fc.chunk_id not in seen:
                    raw_results.append(fc)

        # 4. Filter & Rerank
        filtered_chunks = self.guardrails.filter_relevant_chunks(raw_results, query=cleaned_question)
        if self.citation_boosts and filtered_chunks:
            for c in filtered_chunks:
                boost = self.citation_boosts.get(c.source_file, 0.0)
                if boost:
                    c.score = max(0.0, min(1.0, c.score + boost))

        relevant_chunks = self.guardrails.apply_attention_reranking(
            query=cleaned_question,
            chunks=filtered_chunks,
            history=self.conversation_memory
        )

        if not relevant_chunks:
            if self.missing_answer_behavior == "general_knowledge":
                gen_prompt = (
                    f"Answer the following user question using your general knowledge: {cleaned_question}\n\n"
                    "Requirement: You MUST start your response with: '[General Knowledge - Not found in your documents]'."
                )
                collected = []
                try:
                    for token in MultiProviderLLM.generate_stream(
                        user_prompt=gen_prompt,
                        system_prompt="You are a helpful AI assistant answering from general knowledge because the topic was not found in the user's private documents.",
                        provider=provider_name,
                        model=target_model,
                        api_key=api_key,
                        temperature=0.3,
                        ollama_url=self.ollama.base_url
                    ):
                        collected.append(token)
                        yield {"type": "token", "token": token}
                    full_resp = "".join(collected)
                except Exception:
                    full_resp = f"I could not find any information about '{user_question}' in the uploaded documents."
                    yield {"type": "token", "token": full_resp}
                
                self.conversation_memory.append({"role": "user", "content": user_question})
                self.conversation_memory.append({"role": "assistant", "content": full_resp})
                yield {
                    "type": "done",
                    "answer": full_resp,
                    "is_grounded": False,
                    "confidence_score": 0.25,
                    "citations": [],
                    "images": []
                }
                return
            else:
                fallback_msg = f"I could not find any information about '{user_question}' in the uploaded documents. Please upload a document containing this topic or ask about your indexed files."
                words = fallback_msg.split(" ")
                for i, w in enumerate(words):
                    yield {"type": "token", "token": w + (" " if i < len(words) - 1 else "")}
                self.conversation_memory.append({"role": "user", "content": user_question})
                self.conversation_memory.append({"role": "assistant", "content": fallback_msg})
                yield {
                    "type": "done",
                    "answer": fallback_msg,
                    "is_grounded": False,
                    "confidence_score": 0.0,
                    "citations": [],
                    "images": []
                }
                return

        # 5. Build prompts
        system_prompt = self.guardrails.build_grounded_system_prompt()
        if self.response_style == "concise":
            system_prompt += "\n\nRESPONSE FORMAT DIRECTIVE: Deliver a concise, direct answer using bullet points. Avoid filler introductions or long summaries."
        elif self.response_style == "detailed":
            system_prompt += "\n\nRESPONSE FORMAT DIRECTIVE: Deliver a comprehensive, in-depth explanation with complete context and clear headings."
        feedback_exemplars = self.get_feedback_exemplars(cleaned_question)
        user_prompt = self.guardrails.build_user_prompt(
            query=cleaned_question,
            context_chunks=relevant_chunks,
            conversation_history=self.conversation_memory,
            feedback_exemplars=feedback_exemplars,
            negative_exemplars=negative_exemplars
        )

        # 6. Stream tokens
        collected_tokens = []
        try:
            for token in MultiProviderLLM.generate_stream(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                provider=provider_name,
                model=target_model,
                api_key=api_key,
                temperature=0.1,
                ollama_url=self.ollama.base_url
            ):
                collected_tokens.append(token)
                yield {"type": "token", "token": token}
        except Exception as e:
            err_msg = f"Error communicating with {provider_name.upper()}: {str(e)}"
            yield {"type": "token", "token": err_msg}
            collected_tokens.append(err_msg)

        full_raw = "".join(collected_tokens)
        llm_response = self._sanitize_llm_response(full_raw)

        # Record conversation turn in memory
        self.conversation_memory.append({"role": "user", "content": user_question})
        self.conversation_memory.append({"role": "assistant", "content": llm_response})
        if len(self.conversation_memory) > 12:
            self.conversation_memory = self.conversation_memory[-12:]

        # Evaluate grounding
        is_grounded, confidence_score, grounded_citations = self.guardrails.evaluate_grounding(
            query=cleaned_question,
            answer=llm_response,
            relevant_chunks=relevant_chunks
        )

        matched_images = []
        if is_grounded or is_visual_query:
            matched_images = self._extract_relevant_images(user_question, relevant_chunks, is_image_query=is_visual_query)

        if is_visual_query and not matched_images and hasattr(self.vector_store, "get_image_chunks"):
            img_pool = self.vector_store.get_image_chunks(limit=6)
            if img_pool:
                matched_images = self._extract_relevant_images(user_question, img_pool, is_image_query=True)

        if is_visual_query and matched_images and not is_grounded:
            is_grounded = True
            confidence_score = max(0.65, confidence_score)

        unique_cits = {}
        target_cits = grounded_citations if grounded_citations else relevant_chunks[:3]
        for c in target_cits:
            score_val = float(c.score) if getattr(c, "score", None) is not None else 0.0
            score_pct = round(score_val * 100, 1)
            if c.source_file not in unique_cits or score_pct > unique_cits[c.source_file]["relevance"]:
                unique_cits[c.source_file] = {
                    "source_file": c.source_file,
                    "relevance": score_pct,
                    "text": c.text[:200] + "..." if len(c.text) > 200 else c.text
                }

        yield {
            "type": "done",
            "answer": llm_response,
            "is_grounded": is_grounded,
            "confidence_score": confidence_score,
            "citations": list(unique_cits.values()) if is_grounded else [],
            "images": matched_images if is_grounded else []
        }

    def query_with_image_stream(
        self,
        user_question: str,
        query_image_path: Path | str,
        top_k: int = 6,
        history: Optional[List[Dict[str, str]]] = None,
        provider: str = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Executes a real-time token-by-token streaming Multimodal Visual Search & Query.
        """
        if history is not None:
            self.conversation_memory = history[-10:]

        target_model = model or self.current_model
        provider_name = (provider or "ollama").lower().strip()
        image_path = Path(query_image_path)

        # 1. Perceptual visual fingerprint matching against document extracted figures
        fig_chunks, fig_info, top_image_match = self._find_chunks_for_image_match(image_path)

        # 2. Extract OCR text and Vision description of query image
        vision_parser = getattr(self.parser_factory, "vision_parser", getattr(self.parser_factory, "_image_parser", None))
        if vision_parser:
            image_analysis = vision_parser.describe_and_ocr_image(image_path)
        else:
            image_analysis = {"ocr_text": "", "description": "", "combined_summary": ""}

        ocr_text = image_analysis.get("ocr_text", "")
        vision_desc = image_analysis.get("description", "")
        combined_summary = image_analysis.get("combined_summary", "")

        if fig_info:
            vis_match_block = "\n".join(fig_info)
            combined_summary = f"{vis_match_block}\n\n{combined_summary}" if combined_summary else vis_match_block
        else:
            unmatched_note = (
                "[ATTACHED IMAGE STATUS]: This attached image does NOT match any figure or diagram in the uploaded document(s). "
                "It is an external or distinct image provided by the user."
            )
            combined_summary = f"{unmatched_note}\n\n{combined_summary}" if combined_summary else unmatched_note

        effective_question = user_question.strip() if user_question else "What is this image and how does it relate to the uploaded documents?"
        visual_terms = []
        if fig_info:
            visual_terms.append(" ".join(fig_info))
        if ocr_text:
            clean_ocr = " ".join([w for w in ocr_text.split() if len(w) > 1 or w.isalnum()])
            if clean_ocr:
                visual_terms.append(clean_ocr)
        if vision_desc:
            clean_desc = vision_desc[:300].strip()
            if clean_desc:
                visual_terms.append(clean_desc)

        visual_search_topic = f"{effective_question} {' '.join(visual_terms)}".strip()

        # 3. Retrieve candidates
        retrieval_k = max(top_k * 2, 12)
        raw_results = self.vector_store.query(visual_search_topic, top_k=retrieval_k)

        # Prepend visually matched figure chunks so they are prioritized
        seen_texts = {r.text for r in raw_results}
        for fc in reversed(fig_chunks):
            if fc.text not in seen_texts:
                raw_results.insert(0, fc)
                seen_texts.add(fc.text)

        filtered_chunks = self.guardrails.filter_relevant_chunks(raw_results, query=visual_search_topic)
        filt_texts = {c.text for c in filtered_chunks}
        for fc in reversed(fig_chunks):
            if fc.text not in filt_texts:
                filtered_chunks.insert(0, fc)
                filt_texts.add(fc.text)

        relevant_chunks = self.guardrails.apply_attention_reranking(
            query=visual_search_topic,
            chunks=filtered_chunks,
            history=self.conversation_memory,
            is_image_query=True
        )[:top_k]

        # 4. Construct prompt
        system_prompt = self.guardrails.build_grounded_system_prompt()
        feedback_exemplars = self.get_feedback_exemplars(effective_question)
        user_prompt = self.guardrails.build_user_prompt(
            query=effective_question,
            context_chunks=relevant_chunks,
            user_image_context=combined_summary,
            conversation_history=self.conversation_memory,
            feedback_exemplars=feedback_exemplars
        )

        # 5. Stream tokens
        collected_tokens = []
        try:
            for token in MultiProviderLLM.generate_stream(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                provider=provider_name,
                model=target_model,
                api_key=api_key,
                temperature=0.1,
                ollama_url=self.ollama.base_url
            ):
                collected_tokens.append(token)
                yield {"type": "token", "token": token}
        except Exception as e:
            err_msg = f"Error communicating with {provider_name.upper()}: {str(e)}"
            yield {"type": "token", "token": err_msg}
            collected_tokens.append(err_msg)

        full_raw = "".join(collected_tokens)
        llm_response = self._sanitize_llm_response(full_raw)

        # Record conversation turn
        self.conversation_memory.append({"role": "user", "content": f"[Image Search]: {effective_question}"})
        self.conversation_memory.append({"role": "assistant", "content": llm_response})
        if len(self.conversation_memory) > 12:
            self.conversation_memory = self.conversation_memory[-12:]

        # Grounding
        is_grounded, confidence_score, grounded_citations = self.guardrails.evaluate_grounding(
            query=visual_search_topic,
            answer=llm_response,
            relevant_chunks=relevant_chunks
        )

        if (top_image_match or fig_chunks) and not is_grounded and relevant_chunks:
            is_grounded = True
            confidence_score = max(confidence_score, 0.78)
            grounded_citations = relevant_chunks
        elif not is_grounded and relevant_chunks:
            top_score = max(getattr(c, "score", 0.0) for c in relevant_chunks)
            if top_score >= 0.40:
                is_grounded = True
                confidence_score = max(confidence_score, 0.65)
                grounded_citations = relevant_chunks

        matched_images = []
        if top_image_match:
            matched_images.append(top_image_match)
        if is_grounded or relevant_chunks:
            extra_imgs = self._extract_relevant_images(visual_search_topic, relevant_chunks, is_image_query=True)
            for ei in extra_imgs:
                if not any(m["url"] == ei["url"] for m in matched_images):
                    matched_images.append(ei)
            matched_images = matched_images[:3]
            if matched_images and not is_grounded:
                is_grounded = True
                confidence_score = max(confidence_score, 0.65)
                grounded_citations = relevant_chunks

        unique_cits = {}
        target_cits = grounded_citations if grounded_citations else relevant_chunks[:3]
        for c in target_cits:
            score_val = float(c.score) if getattr(c, "score", None) is not None else 0.0
            score_pct = round(score_val * 100, 1)
            if c.source_file not in unique_cits or score_pct > unique_cits[c.source_file]["relevance"]:
                unique_cits[c.source_file] = {
                    "source_file": c.source_file,
                    "relevance": score_pct,
                    "text": c.text[:200] + "..." if len(c.text) > 200 else c.text
                }

        yield {
            "type": "done",
            "answer": llm_response,
            "is_grounded": is_grounded,
            "confidence_score": confidence_score,
            "citations": list(unique_cits.values()) if is_grounded else [],
            "images": matched_images if is_grounded else []
        }

    @staticmethod
    def _sanitize_llm_response(answer: str) -> str:
        """
        Strips residual AI disclaimers about image display inability or leaked raw API URLs,
        ensuring answers stay clean, professional, and grounded.
        """
        if not answer:
            return answer
        cleaned = answer
        disclaimers = [
            r"(?:[Uu]nfortunately|[Aa]gain|[Oo]nce more)?,?\s*(?:I am|I'm)?\s*unable to display (?:the )?images? directly[^\n.]*[\n.]*",
            r"[Aa]s an AI(?: language model)?,?\s*I cannot (?:display|show) images?[^\n.]*[\n.]*",
            r"[Aa]s a text(?:-based)? AI,?\s*I cannot (?:display|show) images?[^\n.]*[\n.]*",
            r"[Yy]ou can view (?:it|them) by clicking on the provided URL\.?",
            r"[Cc]lick on the provided URL to view (?:it|the image)\.?"
        ]
        for pat in disclaimers:
            cleaned = re.sub(pat, "", cleaned)

        cleaned = re.sub(r"(?:The )?image URL is\s*[`'\"]?/api/sessions/[^`'\"\s)]+[`'\"]?\.?\s*", "", cleaned)
        cleaned = re.sub(r"[`'\"]?/api/sessions/[^`'\"\s)]+[`'\"]?", "", cleaned)
        cleaned = re.sub(r"\s+\.", ".", cleaned)
        cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
        return cleaned.strip()

    def _extract_relevant_images(self, user_question: str, relevant_chunks: list, is_image_query: bool = False) -> list:
        """
        Intelligently determines when visual diagrams should be attached.
        - Suppresses diagrams if user explicitly asked for text-only / summary without diagrams.
        - Strictly verifies that candidate image captions/filenames match the visual subject requested.
        - Prevents attaching unrelated images (e.g. attaching histograms to Grad-CAM or loss function queries).
        """
        if not relevant_chunks:
            return []

        import re
        q_lower = user_question.lower()

        # Explicit negative diagram intent (user asked for text-only summary without images)
        no_diagram_patterns = [
            "without diagram", "without diagrams", "without image", "without images",
            "without figure", "without figures", "no diagram", "no diagrams",
            "no image", "no images", "no visual", "text only", "omit diagram",
            "omit diagrams", "omit image", "omit images", "exclude diagram", "exclude image"
        ]
        if any(p in q_lower for p in no_diagram_patterns):
            return []

        visual_keywords = [
            "diagram", "figure", "chart", "plot", "graph", "architecture", "image", 
            "photo", "picture", "drawing", "illustration", "flowchart", "show me", 
            "look like", "screenshot", "table", "visual", "fig.", "fig ", "workflow",
            "schema"
        ]
        explicit_visual_intent = any(kw in q_lower for kw in visual_keywords)
        is_summary_query = any(w in q_lower for w in ["summary", "summarize", "overview", "what is this document", "what is this paper"])

        # Extract specific subject words from the user query
        q_words = set(re.findall(r"\b\w{3,}\b", q_lower))
        stop_words = {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "this", "that", "these", "those", "is", "are", "was", "were", "give", "tell", "show", "find", "the", "and", "for", "with", "from", "about", "diagram", "diagrams", "image", "images", "figure", "figures", "chart", "visual"}
        subject_words = q_words - stop_words

        # If it's a conceptual question ("what is X", "define X") without visual intent, do not attach images
        is_conceptual = any(q_lower.startswith(p) for p in ["what is ", "what are ", "define ", "how does ", "explain the concept ", "why is "])
        if is_conceptual and not explicit_visual_intent and not is_image_query:
            return []

        scored_images = []
        seen_urls = set()

        for c in relevant_chunks:
            meta_url = c.metadata.get("image_url", "").strip() if isinstance(c.metadata, dict) else ""
            urls_to_check = [meta_url] if meta_url else []

            found_urls = re.findall(r"\[Image URL:\s*(.*?)\]", c.text)
            urls_to_check.extend([u.strip() for u in found_urls if u.strip()])

            cap_match = re.search(r"\[IMAGE / FIGURE:\s*(.*?)\]", c.text, re.IGNORECASE)
            chunk_caption = cap_match.group(1).strip() if cap_match else (c.metadata.get("caption", "") if isinstance(c.metadata, dict) else "")

            for url_clean in urls_to_check:
                if url_clean and url_clean not in seen_urls:
                    seen_urls.add(url_clean)
                    filename = Path(url_clean).name
                    page_num = c.metadata.get("page_number", "") if isinstance(c.metadata, dict) else ""
                    caption = f"{c.source_file}" + (f" (Page {page_num})" if page_num else "")

                    # Calculate semantic relevance between the query and the image
                    img_text = f"{filename} {chunk_caption} {c.text[:1500]}".lower()
                    overlap = sum(1 for w in subject_words if w in img_text) if subject_words else 0

                    # If user asked for a specific visual item (e.g. "Grad-CAM", "ROC curve"), require subject overlap!
                    if explicit_visual_intent and subject_words:
                        if overlap == 0:
                            continue

                    raw_score = getattr(c, "score", 0.75)
                    calibrated_rel = self.guardrails.calibrate_confidence(raw_score, has_lexical_match=(overlap > 0)) * 100.0
                    final_rel = round(max(40.0, calibrated_rel + (overlap * 8)), 1)

                    scored_images.append({
                        "url": url_clean,
                        "filename": filename,
                        "source_file": caption,
                        "relevance": min(100.0, final_rel),
                        "overlap": overlap
                    })

        # Sort images by genuine subject overlap first, then relevance
        scored_images.sort(key=lambda x: (x["overlap"], x["relevance"]), reverse=True)

        # If user did not have explicit visual intent and not a summary query, only attach if overlap > 0
        if not explicit_visual_intent and not is_image_query and not is_summary_query:
            scored_images = [img for img in scored_images if img["overlap"] > 0]

        max_imgs = 3 if (is_image_query or explicit_visual_intent) else 1
        return scored_images[:max_imgs]
