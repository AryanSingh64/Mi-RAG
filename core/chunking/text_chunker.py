from dataclasses import dataclass, field
from typing import Any, Dict, List
from core.ingestion.base import ParsedDocument


@dataclass
class DocumentChunk:
    """
    Represents an atomic, searchable piece of text ready for vector embedding.
    """
    chunk_id: str
    text: str
    source_file: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class RecursiveChunker:
    """
    Production-grade recursive text chunker.
    Splits text hierarchically: Paragraphs -> Sentences -> Words with overlap.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text using the highest-priority separator available."""
        final_chunks = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        good_splits = []
        for s in splits:
            if separator != "":
                s = s + separator if not s.endswith(separator) else s
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits)
                    final_chunks.extend(merged)
                    good_splits = []
                if new_separators:
                    other_splits = self._split_text(s, new_separators)
                    final_chunks.extend(other_splits)
                else:
                    final_chunks.append(s)

        if good_splits:
            merged = self._merge_splits(good_splits)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merges small splits together up to chunk_size while maintaining overlap."""
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_len = len(split)
            if current_length + split_len > self.chunk_size and current_chunk:
                doc_text = "".join(current_chunk).strip()
                if doc_text:
                    chunks.append(doc_text)
                
                # Keep overlap from the end of the current chunk
                while current_length > self.chunk_overlap and current_chunk:
                    popped = current_chunk.pop(0)
                    current_length -= len(popped)

            current_chunk.append(split)
            current_length += split_len

        if current_chunk:
            doc_text = "".join(current_chunk).strip()
            if doc_text:
                chunks.append(doc_text)

        return chunks

    def chunk_document(self, parsed_doc: ParsedDocument) -> List[DocumentChunk]:
        """
        Splits a ParsedDocument into a list of tagged DocumentChunks.
        """
        raw_chunks = self._split_text(parsed_doc.text_content, self.separators)
        document_chunks = []

        for idx, text_chunk in enumerate(raw_chunks, start=1):
            chunk_metadata = {
                "source_file": parsed_doc.filename,
                "file_type": parsed_doc.file_type,
                "chunk_index": idx,
                "total_chunks": len(raw_chunks),
                **parsed_doc.metadata
            }
            
            chunk_id = f"{parsed_doc.filename}_chunk_{idx}"
            
            document_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=text_chunk,
                    source_file=parsed_doc.filename,
                    metadata=chunk_metadata
                )
            )

        return document_chunks
