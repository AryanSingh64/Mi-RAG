"""
Enterprise Multimodal Retrieval-Augmented Generation (RAG) Architecture.
Hardened 6-tier pipeline implementing layout-aware AST extraction, hierarchical parent-child
chunking, unified multimodal embeddings, RRF + Cross-Encoder hybrid retrieval, calibrated NLI
guardrails, and claim-level attribution verification.
"""

from .schemas import (
    DocumentNodeType,
    DocumentNode,
    TableNode,
    FigureNode,
    ParsedDocumentAST,
    ParentChunk,
    ChildChunk,
    RRFResult,
    RerankedCandidate,
    GuardrailEvaluation,
    AtomicClaim,
    AttributionReport,
    GroundedResponse
)

__all__ = [
    "DocumentNodeType",
    "DocumentNode",
    "TableNode",
    "FigureNode",
    "ParsedDocumentAST",
    "ParentChunk",
    "ChildChunk",
    "RRFResult",
    "RerankedCandidate",
    "GuardrailEvaluation",
    "AtomicClaim",
    "AttributionReport",
    "GroundedResponse",
]
