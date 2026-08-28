import shutil
from pathlib import Path
from core.ingestion.base import ParsedDocument
from core.chunking.text_chunker import RecursiveChunker
from core.embeddings.embedder import LocalEmbedder
from core.vectorstore.chroma_store import ChromaVectorStore

def test_pipeline():
    test_db_dir = Path("./data/test_chroma_db")
    if test_db_dir.exists():
        shutil.rmtree(test_db_dir)

    print("1️⃣ Initializing Embedder & ChromaStore...")
    embedder = LocalEmbedder(model_name="all-MiniLM-L6-v2")
    store = ChromaVectorStore(persist_directory=test_db_dir, embedder=embedder)

    # Ingest mock company documents
    sample_text = (
        "Project Orion Launch Date:\n"
        "The public launch for Project Orion is scheduled for November 15, 2026. "
        "The security team must sign off by October 20.\n\n"
        "Budget and Financing:\n"
        "Total allocated budget for marketing is $150,000 USD managed by Sarah Chen."
    )
    doc = ParsedDocument(
        filename="project_orion.txt",
        file_path="/data/project_orion.txt",
        file_type="txt",
        text_content=sample_text,
        metadata={"project": "Orion"}
    )

    print("2️⃣ Chunking document...")
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.chunk_document(doc)
    print(f"   Created {len(chunks)} chunks.")

    print("3️⃣ Adding chunks to Chroma Vector Store...")
    store.add_chunks(chunks)
    print(f"   Indexed total chunks in DB: {store.count()}")

    # Test Query 1: Semantic match (different wording from original text!)
    query_1 = "When are we releasing the Orion software to the public?"
    print(f"\n🔍 Query: \"{query_1}\"")
    results = store.query(query_1, top_k=2)

    for r in results:
        print(f"   🎯 Match [Score: {r.score * 100:.1f}% | File: {r.source_file}]")
        print(f"      Text: \"{r.text}\"")

    # Clean up test DB
    shutil.rmtree(test_db_dir, ignore_errors=True)
    print("\n✅ Vector store & embedding test passed successfully!")

if __name__ == "__main__":
    test_pipeline()
