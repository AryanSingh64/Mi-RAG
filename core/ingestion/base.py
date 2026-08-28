from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ParsedDocument:
    """
    Standardized container representing a parsed document before chunking.
    """
    filename: str
    file_path: str
    file_type: str
    text_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    pages: Optional[List[str]] = None  # Optional per-page text if available


class BaseDocumentParser(ABC):
    """
    Abstract Base Class for all document parsers.
    Every parser must implement the `parse` method.
    """

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parses a file and extracts clean text along with metadata.
        
        Args:
            file_path: Absolute or relative path to the file.
            
        Returns:
            ParsedDocument containing extracted text and metadata.
        """
        pass
