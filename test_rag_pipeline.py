import shutil
import time
from pathlib import Path
from core.pipeline import RAGPipeline

def run_pipeline_test():
    test_db = Path("./data/test_rag_db")
    if test_db.exists():
        shutil.rmtree(test_db)

    print("==================================================")
    print("🚀 Initializing Autonomous RAG Pipeline...")
    print("==================================================")

    pipeline = RAGPipeline(
        persist_directory=test_db,
        ollama_model="llama3.2:1b",
        min_similarity_threshold=0.30
    )

    available_models = pipeline.ollama.list_local_models()
    print(f"📋 Detected Local Ollama Models: {available_models}")

    if "llama3.2:1b" in available_models:
        pipeline.current_model = "llama3.2:1b"
    elif available_models:
        pipeline.current_model = available_models[0]
    
    print(f"🤖 Active Model: {pipeline.current_model}")
    print(f"⚡ CPU Threads In Use: {pipeline.ollama.num_threads} (with RTX 2050 GPU offload)")

    # 1. Ingest sample document
    sample_doc = Path("company_handbook.txt")
    sample_doc.write_text(
        "CYBERSECURITY & REMOTE WORK POLICY 2026\n\n"
        "1. Password Policy: All employee passwords must be at least 16 characters long and must be rotated every 90 days.\n\n"
        "2. Wi-Fi & Network Security: Connecting to public Wi-Fi without the Corporate GlobalProtect VPN is strictly forbidden.\n\n"
        "3. Home Office Tech Allowance: Full-time employees receive a $1,200 annual tech budget for monitors, desks, and ergonomic chairs.\n\n"
        "4. Phishing Incidents: Any suspicious email must be forwarded immediately to security@company.internal within 1 hour.",
        encoding="utf-8"
    )

    print("\n📥 Ingesting Document into Vector DB...")
    chunks_added = pipeline.ingest_file(sample_doc)
    print(f"   Indexed {chunks_added} chunks successfully.\n")

    # List of test questions
    test_questions = [
        "How much money do employees get for home office equipment?",
        "What are the exact rules for passwords?",
        "Can I connect to public Wi-Fi at a coffee shop?",
        "Who is the current Prime Minister of Japan?"  # Out-of-domain Hallucination test
    ]

    for idx, q in enumerate(test_questions, start=1):
        print("--------------------------------------------------")
        print(f"❓ Question {idx}: {q}")
        start_t = time.time()
        ans = pipeline.query(q)
        elapsed = time.time() - start_t

        print(f"\n💡 Answer:\n{ans.answer}\n")
        print(f"⏱️ Response Time: {elapsed:.2f}s")
        print(f"📊 Confidence: {ans.confidence_score * 100:.1f}% | Grounded: {ans.is_grounded}")
        if ans.citations:
            print("📚 Citations:")
            for c in ans.citations:
                print(f"   • {c.source_file} (Relevance: {c.score * 100:.1f}%)")

    # Clean up
    sample_doc.unlink(missing_ok=True)
    shutil.rmtree(test_db, ignore_errors=True)
    print("\n==================================================")
    print("✅ All RAG tests completed successfully!")
    print("==================================================")

if __name__ == "__main__":
    run_pipeline_test()
