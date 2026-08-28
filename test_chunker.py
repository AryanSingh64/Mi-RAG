from core.ingestion.base import ParsedDocument
from core.chunking.text_chunker import RecursiveChunker

def test_chunker():
    print("🚀 Initializing RecursiveChunker (chunk_size=250, chunk_overlap=50)...")
    chunker = RecursiveChunker(chunk_size=250, chunk_overlap=50)

    sample_text = (
        "Acme Corp Leave Policy 2026.\n\n"
        "Employees are entitled to 20 paid vacation days per calendar year. "
        "Vacation requests must be submitted at least 2 weeks in advance through the HR portal.\n\n"
        "Sick leave covers up to 10 days per year with a doctor's certificate required for absences "
        "exceeding 3 consecutive days. Unused sick leave does not roll over to the next year.\n\n"
        "Parental leave provides 12 weeks of fully paid leave for primary caregivers and 4 weeks for secondary caregivers."
    )

    dummy_doc = ParsedDocument(
        filename="company_policy.pdf",
        file_path="/data/company_policy.pdf",
        file_type="pdf",
        text_content=sample_text,
        metadata={"department": "HR", "version": "1.0"}
    )

    chunks = chunker.chunk_document(dummy_doc)

    print(f"\n📄 Original Document Length: {len(sample_text)} characters")
    print(f"✂️ Total Chunks Created: {len(chunks)}\n")

    for idx, c in enumerate(chunks, start=1):
        print(f"--- [Chunk #{idx} | ID: {c.chunk_id}] ---")
        print(f"Content: \"{c.text}\"")
        print(f"Length: {len(c.text)} chars")
        print(f"Metadata: {c.metadata}\n")

    # Assertions / Sanity Checks
    assert len(chunks) > 1, "Should create multiple chunks for long text"
    assert all(c.source_file == "company_policy.pdf" for c in chunks), "Source file should match"
    print("✅ Chunker test passed successfully!")

if __name__ == "__main__":
    test_chunker()
