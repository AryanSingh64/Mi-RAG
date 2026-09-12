"""
Data Models and AST Schemas for Enterprise Multimodal RAG.
Implemented with Pydantic v2, strict typing, and comprehensive relational validation.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class DocumentNodeType(str, Enum):
    SECTION = "section"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    CODE = "code"
    HEADER = "header"
    LIST_ITEM = "list_item"


class BoundingBox(BaseModel):
    """Normalized spatial coordinates on a document page."""
    model_config = ConfigDict(frozen=True)

    page: int = Field(..., description="1-indexed page number")
    left: float = Field(..., ge=0.0, description="Normalized left coordinate [0.0, 1.0]")
    top: float = Field(..., ge=0.0, description="Normalized top coordinate [0.0, 1.0]")
    right: float = Field(..., le=1.01, description="Normalized right coordinate [0.0, 1.0]")
    bottom: float = Field(..., le=1.01, description="Normalized bottom coordinate [0.0, 1.0]")

    @property
    def area(self) -> float:
        return max(0.0, self.right - self.left) * max(0.0, self.bottom - self.top)


class DocumentNode(BaseModel):
    """Atomic node in the document Abstract Syntax Tree (AST)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_id: str = Field(..., description="Unique deterministic identifier (e.g. 'doc1_p3_n4')")
    doc_id: str = Field(..., description="Parent document identifier")
    parent_id: Optional[str] = Field(None, description="Parent AST node ID (e.g. enclosing section)")
    node_type: DocumentNodeType = Field(..., description="Syntactic element classification")
    content: str = Field(..., description="Raw or formatted text content of the node")
    page: int = Field(..., ge=1, description="Source page number")
    bbox: Optional[BoundingBox] = Field(None, description="Spatial coordinates if localized")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Structural and semantic metadata")


class TableNode(DocumentNode):
    """Specialized AST node for structural tables preserving tabular geometry."""
    markdown_repr: str = Field(..., description="Clean Markdown grid representation")
    html_repr: Optional[str] = Field(None, description="Lossless HTML representation with row/col spans")
    num_rows: int = Field(..., ge=1)
    num_cols: int = Field(..., ge=1)
    headers: List[str] = Field(default_factory=list, description="Extracted column headers")


class FigureNode(DocumentNode):
    """Specialized AST node for visual figures, plots, and schematics."""
    image_path: str = Field(..., description="Relative filesystem path to high-res raster crop")
    caption_node_id: Optional[str] = Field(None, description="AST node ID of the bound caption")
    caption_text: str = Field(default="", description="Resolved structural caption text")
    dhash: str = Field(..., description="64-bit difference hash for deduplication only")
    siglip_embedding: Optional[List[float]] = Field(None, description="1152-dim or 768-dim SigLIP visual embedding")
    referenced_in_chunks: List[str] = Field(default_factory=list, description="IDs of linked chunks")


class ParsedDocumentAST(BaseModel):
    """Complete hierarchical Abstract Syntax Tree for a parsed technical document."""
    doc_id: str = Field(..., description="Unique document hash / ID")
    filename: str = Field(..., description="Original filename")
    page_count: int = Field(..., ge=1)
    nodes: List[DocumentNode] = Field(default_factory=list)
    tables: List[TableNode] = Field(default_factory=list)
    figures: List[FigureNode] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChildChunk(BaseModel):
    """Fine-grained chunk optimized for dense vector retrieval (128-256 tokens)."""
    child_id: str = Field(..., description="Unique chunk ID (e.g. 'c_doc1_004')")
    parent_id: str = Field(..., description="Foreign key to ParentChunk")
    doc_id: str = Field(...)
    content: str = Field(..., description="Focused text payload stripped of URL tags")
    token_count: int = Field(..., ge=1)
    section_path: List[str] = Field(default_factory=list, description="Hierarchical section breadcrumbs")
    figure_ids: List[str] = Field(default_factory=list, description="Bound FigureNode IDs")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParentChunk(BaseModel):
    """Comprehensive contextual parent chunk (1024-2048 tokens) injected into LLM context."""
    parent_id: str = Field(..., description="Unique parent ID (e.g. 'p_doc1_001')")
    doc_id: str = Field(...)
    section_title: str = Field(...)
    full_content: str = Field(..., description="Complete multi-paragraph context including tables")
    token_count: int = Field(..., ge=1)
    child_ids: List[str] = Field(default_factory=list)
    figure_metadata: List[Dict[str, Any]] = Field(default_factory=list)


class RRFResult(BaseModel):
    """Consolidated candidate after first-stage Reciprocal Rank Fusion (RRF)."""
    chunk_id: str
    parent_id: str
    doc_id: str
    content: str
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rrf_score: float = Field(..., description="Consolidated RRF score: sum(1 / (k + r))")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RerankedCandidate(BaseModel):
    """Candidate re-scored by cross-encoder for second-stage precision ranking."""
    chunk_id: str
    parent_id: str
    doc_id: str
    content: str
    parent_context: str
    cross_encoder_logit: float
    calibrated_prob: float = Field(..., ge=0.0, le=1.0, description="Sigmoid calibrated relevance probability")
    associated_figures: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GuardrailEvaluation(BaseModel):
    """Multi-tiered validation report evaluating out-of-domain and hallucination risk."""
    passed: bool = Field(..., description="True if generation is authorized to proceed")
    refusal_reason: Optional[str] = None
    cross_encoder_prob: float = Field(..., description="Sigmoid calibrated cross-encoder probability")
    z_score: float = Field(..., description="Corpus-standardized Z-score")
    nli_entailment_prob: float = Field(..., description="Entailment probability of query given context")
    suggested_refusal_message: Optional[str] = None


class AtomicClaim(BaseModel):
    """Discrete atomic factual proposition extracted from the generated text."""
    claim_id: str
    claim_text: str
    is_entailed: bool = Field(..., description="True if supported by retrieved context passages")
    entailment_prob: float = Field(..., ge=0.0, le=1.0)
    supporting_chunk_ids: List[str] = Field(default_factory=list)


class AttributionReport(BaseModel):
    """Formal claim-level faithfulness report (Ragas / TruLens standard)."""
    total_claims: int = Field(..., ge=0)
    supported_claims: int = Field(..., ge=0)
    faithfulness_score: float = Field(..., ge=0.0, le=1.0, description="supported_claims / total_claims")
    claims: List[AtomicClaim] = Field(default_factory=list)
    latency_ms: float = Field(..., description="Verification inference time in milliseconds")


class GroundedResponse(BaseModel):
    """Final enterprise response payload streamed or returned to the client."""
    response_text: str
    referenced_figures: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    guardrails: GuardrailEvaluation
    attribution: Optional[AttributionReport] = None
