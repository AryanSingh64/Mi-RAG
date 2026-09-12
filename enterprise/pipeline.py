"""
Enterprise Multimodal RAG Pipeline Orchestrator.
Integrates all 6 tiers: Layout Ingestion, Parent-Child Chunking, Hybrid RRF Retrieval,
Cross-Encoder Reranking, Calibrated Guardrails, Dynamic Model Routing, and Claim Attribution.
"""

from typing import AsyncGenerator, Dict, List, Optional, Any
from .schemas import (
    ParsedDocumentAST,
    ParentChunk,
    ChildChunk,
    RerankedCandidate,
    GuardrailEvaluation,
    GroundedResponse,
)
from .ingestion import LayoutAwareExtractor
from .chunking import HierarchicalChunker
from .retrieval import HybridRetriever
from .guardrails import CalibratedGuardrails
from .synthesis import EnterpriseSynthesizer, ModelRouter, ClaimLevelAttributionVerifier


class EnterpriseRAGPipeline:
    """
    End-to-End Enterprise Multimodal RAG Engine.
    Replaces prototype heuristics with high-throughput, calibrated information retrieval.
    """

    def __init__(
        self,
        embedder: Any,
        chroma_collection: Any,
        ollama_base_url: str = "http://localhost:11434",
        reranker_model_name: str = "BAAI/bge-reranker-large",
        default_model: str = "qwen2.5:14b-instruct"
    ):
        self.extractor = LayoutAwareExtractor()
        self.chunker = HierarchicalChunker()
        self.parent_store: Dict[str, ParentChunk] = {}
        self.child_store: Dict[str, ChildChunk] = {}

        self.retriever = HybridRetriever(
            dense_embedder=embedder,
            chroma_collection=chroma_collection,
            parent_store=self.parent_store,
            reranker_model_name=reranker_model_name
        )
        self.guardrails = CalibratedGuardrails()
        self.synthesizer = EnterpriseSynthesizer(ollama_base_url=ollama_base_url)
        self.default_model = default_model

    def ingest_file(self, file_path: str) -> ParsedDocumentAST:
        """Executes Tier 1 & Tier 2: Layout AST parsing and Parent-Child chunking."""
        # Tier 1: Layout-Aware Parsing
        ast = self.extractor.parse_document(file_path)

        # Tier 2: Hierarchical Parent-Child Chunking
        parents, children = self.chunker.chunk_ast(ast)

        # Index into in-memory relational stores
        for p in parents:
            self.parent_store[p.parent_id] = p
        for c in children:
            self.child_store[c.child_id] = c

        # Tier 3 & 4: Update BM25 and Vector Index
        self.retriever.update_bm25_index(list(self.child_store.values()))
        return ast

    async def execute_query_stream(
        self,
        query: str,
        has_image_query: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executes Tiers 4, 5, and 6 in an asynchronous streaming pipeline.
        Yields typed SSE event payloads for real-time frontend delivery.
        """
        # Tier 4: Hybrid RRF Retrieval + Cross-Encoder Reranking
        yield {"type": "status", "message": "Executing hybrid retrieval and cross-attention reranking..."}
        candidates = self.retriever.retrieve(query=query, top_candidates=30, final_top_k=5)

        # Tier 5: Calibrated Guardrails & Adaptive Refusal Gates
        yield {"type": "status", "message": "Evaluating calibrated refusal guardrails..."}
        guardrail_eval = self.guardrails.evaluate(query, candidates)

        if not guardrail_eval.passed:
            yield {
                "type": "refusal",
                "message": guardrail_eval.suggested_refusal_message,
                "reason": guardrail_eval.refusal_reason,
                "guardrail": guardrail_eval.model_dump()
            }
            return

        # Tier 6: Dynamic Model Routing & Prompt Assembly
        selected_model = ModelRouter.select_model(
            query=query,
            candidates=candidates,
            has_image_query=has_image_query,
            default_model=self.default_model
        )
        yield {"type": "model_selected", "model": selected_model}

        # Resolve associated diagrams
        _, resolved_figures = self.synthesizer.assemble_prompt(query, candidates)
        for fig in resolved_figures:
            yield {"type": "figure", "figure": fig}

        # Stream generation
        full_response_text = ""
        async for token in self.synthesizer.stream_synthesis(query, candidates, selected_model):
            full_response_text += token
            yield {"type": "token", "token": token}

        # Post-Generation Claim-Level Faithfulness Verification
        yield {"type": "status", "message": "Verifying claim-level factual attribution..."}
        contexts = [c.parent_context for c in candidates]
        attribution_report = self.synthesizer.verifier.verify_faithfulness(full_response_text, contexts)

        yield {
            "type": "attribution",
            "report": attribution_report.model_dump()
        }
        yield {"type": "done"}
