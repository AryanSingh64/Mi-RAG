from pathlib import Path
from core.ingestion.base import BaseDocumentParser, ParsedDocument


class TextDocumentParser(BaseDocumentParser):
    """
    Parser for text-based formats (.txt, .md, .csv, .json).
    """

    def parse(self, file_path: Path) -> ParsedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read with UTF-8 encoding, fallback gracefully to latin-1
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin-1", errors="replace")

        ext = file_path.suffix.lower().lstrip(".")
        
        return ParsedDocument(
            filename=file_path.name,
            file_path=str(file_path.resolve()),
            file_type=ext,
            text_content=content.strip(),
            metadata={
                "char_count": len(content),
                "extension": ext
            }
        )
