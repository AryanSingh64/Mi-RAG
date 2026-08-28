from pathlib import Path
from docx import Document
from core.ingestion.base import BaseDocumentParser, ParsedDocument


class DocxDocumentParser(BaseDocumentParser):
    """
    Parser for Microsoft Word (.docx) files.
    Extracts structured paragraphs and table cells.
    """

    def parse(self, file_path: Path) -> ParsedDocument:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc = Document(str(file_path))
        extracted_sections = []

        # 1. Extract body paragraphs (headings and text)
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                extracted_sections.append(text)

        # 2. Extract tables (tabular enterprise data)
        for table_idx, table in enumerate(doc.tables, start=1):
            table_lines = [f"\n--- Table {table_idx} ---"]
            for row in table.rows:
                row_cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                table_lines.append(" | ".join(row_cells))
            if len(table_lines) > 1:
                extracted_sections.append("\n".join(table_lines))

        full_text = "\n\n".join(extracted_sections)

        return ParsedDocument(
            filename=file_path.name,
            file_path=str(file_path.resolve()),
            file_type="docx",
            text_content=full_text,
            metadata={
                "paragraph_count": len(doc.paragraphs),
                "table_count": len(doc.tables),
                "char_count": len(full_text),
            }
        )
