"""
Hierarchical Parent-Child Chunking and Late Chunking Engine.
Preserves contextual integrity without semantic fragmentation or embedding token dilution.
"""

import uuid
import re
from typing import Dict, List, Tuple, Optional, Any
from .schemas import (
    ParsedDocumentAST,
    DocumentNode,
    DocumentNodeType,
    TableNode,
    FigureNode,
    ChildChunk,
    ParentChunk,
)


def _approximate_tokens(text: str) -> int:
    """Fast, accurate token estimator (average 4 chars per token for English/technical prose)."""
    return max(1, len(text) // 4)


class HierarchicalChunker:
    """
    Enterprise Context-Preserving Chunker.
    Generates high-density child chunks (128-256 tokens) for vector indexing
    coupled to comprehensive parent chunks (1024-2048 tokens) for LLM context assembly.
    Metadata (URLs, figure references, table schemas) is strictly isolated from vector payloads.
    """

    def __init__(
        self,
        child_target_tokens: int = 192,
        child_overlap_tokens: int = 32,
        parent_target_tokens: int = 1536
    ):
        self.child_target = child_target_tokens
        self.child_overlap = child_overlap_tokens
        self.parent_target = parent_target_tokens

    def chunk_ast(self, ast: ParsedDocumentAST) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        """
        Processes an AST into coupled ParentChunk and ChildChunk sets.
        Preserves structural tables and maps associated figures via metadata IDs.
        """
        parent_chunks: List[ParentChunk] = []
        child_chunks: List[ChildChunk] = []

        # 1. Group AST nodes into logical parent sections (1024-2048 tokens)
        current_section_title = f"Document: {ast.filename}"
        current_parent_nodes: List[DocumentNode] = []
        current_parent_tokens = 0
        parent_counter = 0

        # Pre-index figures by page for relational linking
        figures_by_page: Dict[int, List[FigureNode]] = {}
        for fig in ast.figures:
            figures_by_page.setdefault(fig.page, []).append(fig)

        def flush_parent() -> Optional[ParentChunk]:
            nonlocal current_parent_nodes, current_parent_tokens, parent_counter
            if not current_parent_nodes:
                return None

            parent_id = f"p_{ast.doc_id}_{parent_counter:04d}"
            parent_counter += 1

            # Assemble full parent content text
            content_parts = []
            pages_covered = set()
            for n in current_parent_nodes:
                pages_covered.add(n.page)
                if isinstance(n, TableNode):
                    content_parts.append(f"\n[Table Layout]:\n{n.markdown_repr}\n")
                else:
                    content_parts.append(n.content)

            full_parent_text = "\n\n".join(content_parts)
            parent_token_count = _approximate_tokens(full_parent_text)

            # Collect linked figure metadata
            linked_figs = []
            for p in pages_covered:
                for f in figures_by_page.get(p, []):
                    linked_figs.append({
                        "node_id": f.node_id,
                        "image_path": f.image_path,
                        "caption": f.caption_text,
                        "page": f.page
                    })

            parent_obj = ParentChunk(
                parent_id=parent_id,
                doc_id=ast.doc_id,
                section_title=current_section_title,
                full_content=full_parent_text,
                token_count=parent_token_count,
                child_ids=[],
                figure_metadata=linked_figs
            )

            # Generate child chunks exclusively inside this parent's scope
            created_children = self._generate_children(parent_obj, current_parent_nodes)
            parent_obj.child_ids = [c.child_id for c in created_children]

            parent_chunks.append(parent_obj)
            child_chunks.extend(created_children)

            # Reset
            current_parent_nodes = []
            current_parent_tokens = 0
            return parent_obj

        for node in ast.nodes:
            # Check for header-driven section boundaries
            if node.node_type == DocumentNodeType.HEADER:
                if current_parent_tokens >= self.child_target * 2:
                    flush_parent()
                current_section_title = node.content.strip()

            node_tokens = _approximate_tokens(node.content)
            if current_parent_tokens + node_tokens > self.parent_target and current_parent_nodes:
                flush_parent()

            current_parent_nodes.append(node)
            current_parent_tokens += node_tokens

        flush_parent()
        return parent_chunks, child_chunks

    def _generate_children(self, parent: ParentChunk, nodes: List[DocumentNode]) -> List[ChildChunk]:
        """
        Splits parent contents into concise child chunks (128-256 tokens).
        Ensures NO synthetic URLs or image tag strings dilute the text payload.
        """
        children: List[ChildChunk] = []
        child_counter = 0

        # Accumulate paragraphs into semantic units
        buffer_paragraphs: List[str] = []
        buffer_tokens = 0
        bound_figure_ids: set[str] = set()

        for node in nodes:
            # Associate figures on this page
            for fig_meta in parent.figure_metadata:
                if fig_meta.get("page") == node.page:
                    bound_figure_ids.add(fig_meta["node_id"])

            para_text = node.content.strip()
            if not para_text:
                continue

            tokens = _approximate_tokens(para_text)

            # If a single paragraph exceeds child limit, perform clean sentence splitting
            if tokens > self.child_target:
                sentences = re.split(r'(?<=[.?!])\s+', para_text)
                for sent in sentences:
                    stok = _approximate_tokens(sent)
                    if buffer_tokens + stok > self.child_target and buffer_paragraphs:
                        child_id = f"c_{parent.parent_id}_{child_counter:03d}"
                        child_counter += 1
                        payload = " ".join(buffer_paragraphs)
                        children.append(ChildChunk(
                            child_id=child_id,
                            parent_id=parent.parent_id,
                            doc_id=parent.doc_id,
                            content=payload,
                            token_count=buffer_tokens,
                            section_path=[parent.section_title],
                            figure_ids=list(bound_figure_ids),
                            metadata={"doc_id": parent.doc_id}
                        ))
                        # Maintain overlap
                        buffer_paragraphs = buffer_paragraphs[-1:] if self.child_overlap > 0 else []
                        buffer_tokens = sum(_approximate_tokens(p) for p in buffer_paragraphs)

                    buffer_paragraphs.append(sent)
                    buffer_tokens += stok
            else:
                if buffer_tokens + tokens > self.child_target and buffer_paragraphs:
                    child_id = f"c_{parent.parent_id}_{child_counter:03d}"
                    child_counter += 1
                    payload = " ".join(buffer_paragraphs)
                    children.append(ChildChunk(
                        child_id=child_id,
                        parent_id=parent.parent_id,
                        doc_id=parent.doc_id,
                        content=payload,
                        token_count=buffer_tokens,
                        section_path=[parent.section_title],
                        figure_ids=list(bound_figure_ids),
                        metadata={"doc_id": parent.doc_id}
                    ))
                    buffer_paragraphs = buffer_paragraphs[-1:] if self.child_overlap > 0 else []
                    buffer_tokens = sum(_approximate_tokens(p) for p in buffer_paragraphs)

                buffer_paragraphs.append(para_text)
                buffer_tokens += tokens

        if buffer_paragraphs:
            child_id = f"c_{parent.parent_id}_{child_counter:03d}"
            payload = " ".join(buffer_paragraphs)
            children.append(ChildChunk(
                child_id=child_id,
                parent_id=parent.parent_id,
                doc_id=parent.doc_id,
                content=payload,
                token_count=buffer_tokens,
                section_path=[parent.section_title],
                figure_ids=list(bound_figure_ids),
                metadata={"doc_id": parent.doc_id}
            ))

        return children


class LateChunkingPooler:
    """
    Jina AI Late Chunking Contextual Pooling Engine.
    Executes a single forward pass over an 8k document context, then mean-pools
    token hidden states over chunk boundaries (s_k, e_k) to maintain global attention.
    """

    @staticmethod
    def pool_chunk_embeddings(
        token_embeddings: Any,
        chunk_token_spans: List[Tuple[int, int]]
    ) -> List[List[float]]:
        """
        Pools contextualized token embeddings over chunk boundary intervals [start, end].
        v_k = 1 / (e_k - s_k + 1) * sum(h_i)
        """
        import numpy as np

        pooled_vectors: List[List[float]] = []
        for start_idx, end_idx in chunk_token_spans:
            if start_idx >= end_idx or start_idx >= len(token_embeddings):
                span_emb = token_embeddings[min(start_idx, len(token_embeddings) - 1)]
            else:
                span_emb = np.mean(token_embeddings[start_idx:end_idx], axis=0)

            # L2 normalize
            norm = np.linalg.norm(span_emb)
            if norm > 1e-12:
                span_emb = span_emb / norm
            pooled_vectors.append(span_emb.tolist())

        return pooled_vectors
