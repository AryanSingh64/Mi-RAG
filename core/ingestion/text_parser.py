import math
from pathlib import Path
from typing import Any, Optional
from core.ingestion.base import BaseDocumentParser, ParsedDocument


class TextDocumentParser(BaseDocumentParser):
    """
    Parser for text-based formats (.txt, .md, .csv, .json, .log).
    Supports two-way page range slicing (~55 lines / 3000 chars per standard page).
    """

    LINES_PER_PAGE = 55
    CHARS_PER_PAGE = 3000

    def parse(
        self,
        file_path: Path,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        progress_callback: Optional[Any] = None
    ) -> ParsedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read with UTF-8 encoding, fallback gracefully to latin-1
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin-1", errors="replace")

        ext = file_path.suffix.lower().lstrip(".")
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)
        total_chars = len(content)

        # Estimate standard pages based on lines or chars
        total_doc_pages = max(1, max(math.ceil(total_chars / self.CHARS_PER_PAGE), math.ceil(total_lines / self.LINES_PER_PAGE)))

        if start_page is not None or end_page is not None:
            s_p = max(1, start_page) if start_page is not None else 1
            e_p = min(total_doc_pages, end_page) if end_page is not None else total_doc_pages
            if s_p > e_p:
                s_p, e_p = 1, total_doc_pages

            # Calculate line slice
            start_line_idx = max(0, (s_p - 1) * self.LINES_PER_PAGE)
            end_line_idx = min(total_lines, e_p * self.LINES_PER_PAGE)
            if start_line_idx < total_lines:
                sliced_lines = lines[start_line_idx:end_line_idx]
                content = "".join(sliced_lines)
            active_pages = max(1, e_p - s_p + 1)
        else:
            s_p, e_p = 1, total_doc_pages
            active_pages = total_doc_pages

        return ParsedDocument(
            filename=file_path.name,
            file_path=str(file_path.resolve()),
            file_type=ext,
            text_content=content.strip(),
            metadata={
                "char_count": len(content),
                "extension": ext,
                "total_pages": active_pages,
                "total_doc_pages": total_doc_pages,
                "page_range": f"{s_p}-{e_p}",
            }
        )
