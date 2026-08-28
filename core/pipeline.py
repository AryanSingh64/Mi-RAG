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
        chunk_size: int = 400,
        chunk_overlap: int = 60,
        min_similarity_threshold: float = 0.08
    ):
        self.vision_models = vision_models or ["moondream"]
        self.parser_factory = DocumentParserFactory(vision_models=self.vision_models)
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

        return GroundedAnswer(
            answer=llm_response,
            is_grounded=True,
            confidence_score=round(avg_confidence, 4),
            citations=relevant_chunks
        )
