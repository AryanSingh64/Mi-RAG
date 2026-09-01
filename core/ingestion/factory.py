from pathlib import Path
from typing import Any, List, Optional, Union
from core.ingestion.base import BaseDocumentParser, ParsedDocument
from core.ingestion.docx_parser import DocxDocumentParser
from core.ingestion.pdf_parser import PdfDocumentParser
from core.ingestion.text_parser import TextDocumentParser
from core.ingestion.vision_ocr import VisionImageParser


class DocumentParserFactory:
    """
    Unified factory to parse any supported document type with multi-model Vision ensemble support
    and automatic multimodal diagram extraction.
    """

    def __init__(
        self,
        vision_models: Optional[Union[List[str], str]] = None,
        output_images_dir: Optional[Path] = None,
        session_id: Optional[str] = None
    ):
        self.vision_models = vision_models or ["moondream"]
        self.output_images_dir = Path(output_images_dir) if output_images_dir else None
        self.session_id = session_id

        self._text_parser = TextDocumentParser()
        self._docx_parser = DocxDocumentParser()
        self._pdf_parser = PdfDocumentParser(
            output_images_dir=self.output_images_dir,
            session_id=self.session_id
        )
        self._image_parser = VisionImageParser(
            vision_models=self.vision_models,
            output_images_dir=self.output_images_dir,
            session_id=self.session_id
        )
        self.vision_parser = self._image_parser
        self.image_parser = self._image_parser

    def parse_file(self, file_path: Path | str, progress_callback: Optional[Any] = None) -> ParsedDocument:
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
            return self._pdf_parser.parse(path, progress_callback=progress_callback)
        elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            return self._image_parser.parse(path)
        else:
            return self._text_parser.parse(path)
