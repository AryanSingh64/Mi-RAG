from pathlib import Path
from typing import List, Optional
from core.chunking.text_chunker import RecursiveChunker
from core.embeddings.embedder import LocalEmbedder
from core.guardrails.anti_hallucination import AntiHallucinationEngine, GroundedAnswer
from core.guardrails.query_rewriter import QueryRewriter
from core.ingestion.factory import DocumentParserFactory
from core.llm.ollama_client import OllamaClient
from core.vectorstore.chroma_store import ChromaVectorStore


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
        min_similarity_threshold: float = 0.08
    ):
        self.session_id = session_id
        self.extracted_images_dir = Path(extracted_images_dir) if extracted_images_dir else None
        self.vision_models = vision_models or ["moondream"]
        self.parser_factory = DocumentParserFactory(
            vision_models=self.vision_models,
            output_images_dir=self.extracted_images_dir,
            session_id=self.session_id
        )
        self.chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = LocalEmbedder(model_name=embedding_model)
        self.vector_store = ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedder=self.embedder
        )
        self.ollama = OllamaClient(base_url=ollama_url, default_model=ollama_model)
        self.guardrails = AntiHallucinationEngine(min_similarity_threshold=min_similarity_threshold)
        self.rewriter = QueryRewriter(ollama_client=self.ollama)
        self.current_model = ollama_model

    def ingest_file(self, file_path: Path | str) -> int:
        """
        Parses a file (text, docx, PDF with images/OCR, standalone images),
        chunks it, and indexes it into ChromaDB.
        """
        parsed_doc = self.parser_factory.parse_file(file_path)
        chunks = self.chunker.chunk_document(parsed_doc)
        self.vector_store.add_chunks(chunks)
        return len(chunks)

    def query(self, user_question: str, top_k: int = 6) -> GroundedAnswer:
        """
        Executes a query with spelling auto-correction, query expansion, and anti-hallucination.
        """
        # 1. Handle casual greetings ("hi", "hello") politely
        greeting_reply = self.rewriter.is_conversational_greeting(user_question)
        if greeting_reply:
            return GroundedAnswer(
                answer=greeting_reply,
                is_grounded=True,
                confidence_score=1.0,
                citations=[]
            )

        # 2. Fix typos & expand query
        cleaned_question, search_query = self.rewriter.clean_and_expand_query(
            user_question,
            model_name=self.current_model
        )

        # 3. Multi-strategy retrieval (corrected query + original query)
        raw_results = self.vector_store.query(search_query, top_k=top_k)
        if search_query != user_question:
            direct_results = self.vector_store.query(user_question, top_k=top_k)
            # Merge and deduplicate by text/source
            seen = {r.text: r for r in raw_results}
            for dr in direct_results:
                if dr.text not in seen:
                    raw_results.append(dr)

        # 4. Filter relevant chunks using guardrails
        relevant_chunks = self.guardrails.filter_relevant_chunks(raw_results)

        # Print retrieved context live in terminal
        print("\n" + "="*60)
        print(f"[*] QUERY: {user_question}")
        if relevant_chunks:
            print(f"[*] RETRIEVED CONTEXT ({len(relevant_chunks)} chunks):")
            for idx, chunk in enumerate(relevant_chunks, 1):
                preview = chunk.text.replace("\n", " ")[:140]
                print(f"  [{idx}] {chunk.source_file} (Similarity: {chunk.score*100:.1f}%) -> {preview}...")
        else:
            print("[!] No relevant chunks met the similarity threshold.")
        print("="*60 + "\n")

        if not relevant_chunks:
            return GroundedAnswer(
                answer="I could not find any information about this in the uploaded documentation.",
                is_grounded=False,
                confidence_score=0.0,
                citations=[],
                warning="No relevant chunks met the threshold."
            )

        # 6. Construct grounded prompt with the CLEANED question
        system_prompt = self.guardrails.build_grounded_system_prompt()
        user_prompt = self.guardrails.build_user_prompt(cleaned_question, relevant_chunks)

        # 7. Query Ollama with low temperature
        try:
            llm_response = self.ollama.chat_response(
                user_message=user_prompt,
                system_prompt=system_prompt,
                model=self.current_model,
                temperature=0.1
            )
        except Exception as e:
            return GroundedAnswer(
                answer=f"Error communicating with local Ollama: {str(e)}",
                is_grounded=False,
                confidence_score=0.0,
                citations=relevant_chunks,
                warning="Ollama connection failed. Is Ollama running?"
            )

        avg_confidence = sum(c.score for c in relevant_chunks) / len(relevant_chunks)

        # Extract only visually relevant diagrams (prevents dumping images on text/summary queries)
        matched_images = self._extract_relevant_images(user_question, relevant_chunks, is_image_query=False)

        if matched_images:
            print(f"[*] ATTACHED VISUAL DIAGRAMS ({len(matched_images)}):")
            for img in matched_images:
                print(f"    - {img['source_file']} -> {img['url']}")

        return GroundedAnswer(
            answer=llm_response,
            is_grounded=True,
            confidence_score=round(avg_confidence, 4),
            citations=relevant_chunks,
            images=matched_images
        )

    def query_with_image(
        self,
        user_question: str,
        query_image_path: Path | str,
        top_k: int = 6
    ) -> GroundedAnswer:
        """
        Executes a Multimodal Visual Search & Query.
        Analyzes the user's attached image (OCR text + Vision scene analysis), searches the knowledge base,
        and provides a grounded comparative answer with document citations and diagrams.
        """
        image_path = Path(query_image_path)
        print(f"\n[*] MULTIMODAL QUERY WITH ATTACHED IMAGE: {image_path.name}")

        # 1. Extract OCR text and Vision description of query image
        vision_parser = getattr(self.parser_factory, "vision_parser", getattr(self.parser_factory, "_image_parser", None))
        if vision_parser:
            image_analysis = vision_parser.describe_and_ocr_image(image_path)
        else:
            image_analysis = {"ocr_text": "", "description": "", "combined_summary": ""}

        ocr_text = image_analysis.get("ocr_text", "")
        vision_desc = image_analysis.get("description", "")
        combined_summary = image_analysis.get("combined_summary", "")

        # 2. Formulate enriched search query
        effective_question = user_question.strip() if user_question else "What is this image and how does it relate to the uploaded documents?"
        
        search_terms = [effective_question]
        if ocr_text:
            search_terms.append(ocr_text)
        if vision_desc:
            search_terms.append(vision_desc[:300])
        
        compound_search_query = " ".join(search_terms)

        # 3. Retrieve relevant chunks from ChromaDB
        raw_results = self.vector_store.query(compound_search_query, top_k=top_k)
        if effective_question != compound_search_query:
            direct_results = self.vector_store.query(effective_question, top_k=top_k)
            seen = {r.text: r for r in raw_results}
            for dr in direct_results:
                if dr.text not in seen:
                    raw_results.append(dr)

        relevant_chunks = self.guardrails.filter_relevant_chunks(raw_results)

        # 4. Construct prompt with user image analysis
        system_prompt = self.guardrails.build_grounded_system_prompt()
        user_prompt = self.guardrails.build_user_prompt(
            query=effective_question,
            context_chunks=relevant_chunks,
            user_image_context=combined_summary
        )

        # 5. Query Ollama
        try:
            llm_response = self.ollama.chat_response(
                user_message=user_prompt,
                system_prompt=system_prompt,
                model=self.current_model,
                temperature=0.1
            )
        except Exception as e:
            return GroundedAnswer(
                answer=f"Error communicating with local Ollama: {str(e)}",
                is_grounded=False,
                confidence_score=0.0,
                citations=relevant_chunks,
                warning="Ollama connection failed. Is Ollama running?"
            )

        avg_confidence = sum(c.score for c in relevant_chunks) / len(relevant_chunks) if relevant_chunks else 0.0

        # 6. Extract top matching document diagrams (capped at 2-3 most relevant)
        matched_images = self._extract_relevant_images(effective_question, relevant_chunks, is_image_query=True)

        return GroundedAnswer(
            answer=llm_response,
            is_grounded=True,
            confidence_score=round(avg_confidence, 4),
            citations=relevant_chunks,
            images=matched_images
        )

    def _extract_relevant_images(self, user_question: str, relevant_chunks: list, is_image_query: bool = False) -> list:
        """
        Intelligently determines when visual diagrams should be attached.
        - Text summaries and general questions return NO images.
        - Attaches diagrams ONLY if the question explicitly mentions visual artifacts
          (e.g., 'diagram', 'figure', 'chart', 'plot', 'image', 'graph', 'architecture', 'photo', 'show me')
          OR if this is a Multimodal Visual Search query.
        - Caps returned images to the top 2-3 highest-confidence matches.
        """
        if not relevant_chunks:
            return []

        q_lower = user_question.lower()
        visual_keywords = [
            "diagram", "figure", "chart", "plot", "graph", "architecture", "image", 
            "photo", "picture", "drawing", "illustration", "flowchart", "show me", 
            "look like", "screenshot", "table", "visual", "fig.", "fig "
        ]
        explicit_visual_intent = any(kw in q_lower for kw in visual_keywords)

        # Pure text queries and summaries return text only
        if not is_image_query and not explicit_visual_intent:
            has_high_diag = any(
                ("[DIAGRAM" in c.text or "[Exact OCR" in c.text) and c.score >= 0.45
                for c in relevant_chunks
            )
            if not has_high_diag:
                return []

        matched_images = []
        seen_urls = set()
        import re

        for c in relevant_chunks:
            meta_url = c.metadata.get("image_url", "").strip() if isinstance(c.metadata, dict) else ""
            urls_to_check = [meta_url] if meta_url else []

            found_urls = re.findall(r"\[Image URL:\s*(.*?)\]", c.text)
            urls_to_check.extend([u.strip() for u in found_urls if u.strip()])

            for url_clean in urls_to_check:
                if url_clean and url_clean not in seen_urls:
                    is_overview = "_page_" in url_clean.lower()
                    if is_overview and not is_image_query and not explicit_visual_intent:
                        continue

                    seen_urls.add(url_clean)
                    filename = Path(url_clean).name
                    page_num = c.metadata.get("page_number", "") if isinstance(c.metadata, dict) else ""
                    caption = f"{c.source_file}" + (f" (Page {page_num})" if page_num else "")
                    matched_images.append({
                        "url": url_clean,
                        "filename": filename,
                        "source_file": caption,
                        "relevance": round(c.score * 100, 1)
                    })

                    if len(matched_images) >= (3 if is_image_query else 2):
                        break

            if len(matched_images) >= (3 if is_image_query else 2):
                break

        return matched_images
