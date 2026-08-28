from pathlib import Path
from typing import Dict, Optional, Type
from core.ingestion.base import BaseDocumentParser, ParsedDocument
from core.ingestion.docx_parser import DocxDocumentParser
from core.ingestion.pdf_parser import PdfDocumentParser
from core.ingestion.text_parser import TextDocumentParser
from core.ingestion.vision_ocr import VisionImageParser


class DocumentParserFactory:
    """
    Unified factory to parse any supported document type with dynamic Vision model support.
    """

    def __init__(self, vision_model: Optional[str] = "moondream"):
        self.vision_model = vision_model
        self._text_parser = TextDocumentParser()
        self._docx_parser = DocxDocumentParser()
        self._pdf_parser = PdfDocumentParser()
        self._image_parser = VisionImageParser(vision_model=self.vision_model or "moondream")

    def parse_file(self, file_path: Path | str) -> ParsedDocument:
        """
        Identifies the file type and routes it to the corresponding parser.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in [".txt", ".md", ".csv", ".json", ".log"]:
            return self._text_parser.parse(path)
        elif ext in [".docx"]:
            return self._docx_parser.parse(path)
        elif ext in [".pdf"]:
            return self._pdf_parser.parse(path)
        elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            return self._image_parser.parse(path)
        else:
            # Fallback: attempt text parsing
            return self._text_parser.parse(path)
