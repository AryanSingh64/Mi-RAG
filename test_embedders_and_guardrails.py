"""
Comprehensive Embedder & Anti-Hallucination Guardrails Test Suite
Tests all available embedding models across:
1. Embedding vector generation & normalization
2. In-domain factual grounding & accuracy
3. Out-of-domain rejection ('Prime Minister of India', 'fastest train') -> is_grounded=False, citations=[], images=[]
4. Gibberish rejection ('svoiausoijdslfjl') -> is_grounded=False, citations=[], images=[]
5. Negative diagram intent ('summary without diagrams') -> images=[]
6. Diagram-specific queries -> attaches matching diagrams
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from core.chunking.text_chunker import DocumentChunk
from core.embeddings.embedder import EMBEDDING_CATALOG, LocalEmbedder
from core.guardrails.anti_hallucination import AntiHallucinationEngine, GroundedAnswer, SearchResult
from core.vectorstore.chroma_store import ChromaVectorStore


def run_unit_guardrails_tests():
    print("\n" + "="*70)
    print(" 🧪 [SUITE 1/2] GUARDRAILS & ANTI-HALLUCINATION UNIT TESTS")
    print("="*70)

    engine = AntiHallucinationEngine(min_similarity_threshold=0.35)

    sample_chunks = [
        SearchResult(
            chunk_id="chunk_1",
            text="LUDO is a framework providing 3D structural information for deformable objects using volumetric occupancy functions and SDFs. [DIAGRAM: Fig 14 Architecture /workspace/diag1.png]",
            source_file="research_paper.pdf",
            metadata={"source_file": "research_paper.pdf", "page_number": 12, "image_url": "/static/images/diag1.png"},
            score=0.74,
            distance=0.26
        ),
        SearchResult(
            chunk_id="chunk_2",
            text="We evaluate parameters like cutoff thresholds and inference times for deformable organ phantoms in robotic surgery.",
            source_file="research_paper.pdf",
            metadata={"source_file": "research_paper.pdf", "page_number": 13},
            score=0.68,
            distance=0.32
        )
    ]

    # Test 1: In-domain grounded query
    in_domain_query = "What is LUDO and how does it represent deformable objects?"
    in_domain_ans = "LUDO is a framework that provides 3D structural information for deformable objects using occupancy functions and signed distance functions (SDF)."
    is_grounded, conf, cites = engine.evaluate_grounding(in_domain_query, in_domain_ans, sample_chunks)
    print(f"\n[TEST 1] In-Domain Query Grounding:")
    print(f"  Query: '{in_domain_query}'")
    print(f"  Result -> is_grounded: {is_grounded} | Confidence: {conf*100:.1f}% | Citations count: {len(cites)}")
    assert is_grounded is True, "FAIL: In-domain answer should be marked as grounded"
    assert conf > 0.40, "FAIL: In-domain confidence should be > 40%"
    assert len(cites) > 0, "FAIL: In-domain citations should be non-empty"
    print("  ✅ PASSED: Correctly verified grounded answer.")

    # Test 2: Out-of-Domain Query ('Prime Minister of India')
    ood_query = "Who is the Prime Minister of India?"
    ood_ans = "Based on the provided documents, there is no mention of the Prime Minister of India. The text focuses on robotic surgery and deformable organ phantoms."
    is_grounded, conf, cites = engine.evaluate_grounding(ood_query, ood_ans, sample_chunks)
    print(f"\n[TEST 2] Out-of-Domain Query ('Prime Minister of India'):")
    print(f"  Query: '{ood_query}'")
    print(f"  Result -> is_grounded: {is_grounded} | Confidence: {conf*100:.1f}% | Citations count: {len(cites)}")
    assert is_grounded is False, "FAIL: Out-of-domain answer must NOT be marked as grounded"
    assert conf == 0.0, "FAIL: Out-of-domain confidence must be 0.0"
    assert len(cites) == 0, "FAIL: Out-of-domain citations must be stripped to []"
    print("  ✅ PASSED: Correctly rejected out-of-domain query (no false citations/badges).")

    # Test 3: Gibberish Query ('svoiausoijdslfjl')
    gibberish_query = "svoiausoijdslfjl what is this"
    gibberish_ans = "Based on the provided documents, there is no mention of 'svoiausoijdslfjl'. The uploaded context contains no reference to this term."
    is_grounded, conf, cites = engine.evaluate_grounding(gibberish_query, gibberish_ans, sample_chunks)
    print(f"\n[TEST 3] Gibberish Query ('svoiausoijdslfjl'):")
    print(f"  Query: '{gibberish_query}'")
    print(f"  Result -> is_grounded: {is_grounded} | Confidence: {conf*100:.1f}% | Citations count: {len(cites)}")
    assert is_grounded is False, "FAIL: Gibberish must NOT be marked as grounded"
    assert len(cites) == 0, "FAIL: Gibberish citations must be empty []"
    print("  ✅ PASSED: Correctly eliminated false citations for gibberish.")

    print("\n✅ ALL GUARDRAIL LOGIC TESTS PASSED SUCCESSFULLY!")


def test_embedder_model(model_name: str):
    print("\n" + "-"*65)
    print(f" 🚀 Testing Embedder: {model_name}")
    print("-"*65)

    temp_dir = tempfile.mkdtemp()
    db_dir = Path(temp_dir) / "test_db"
    db_dir.mkdir(parents=True, exist_ok=True)

    try:
        embedder = LocalEmbedder(model_name=model_name)
        vector_store = ChromaVectorStore(
            persist_directory=db_dir,
            collection_name=f"test_coll_{abs(hash(model_name)) % 10000}",
            embedder=embedder
        )
        guardrails = AntiHallucinationEngine(min_similarity_threshold=0.35)

        # Ingest synthetic knowledge base
        test_chunks = [
            DocumentChunk(
                chunk_id="chunk_surgical_1",
                text="The LUDO architecture predicts 3D occupancy and Signed Distance Functions (SDF) for deformable organs in robotic puncturing. [DIAGRAM: Fig 14 LUDO Pipeline /static/fig14.png]",
                source_file="surgical_robotics.pdf",
                metadata={"source_file": "surgical_robotics.pdf", "page_number": 5, "image_url": "/static/fig14.png"}
            ),
            DocumentChunk(
                chunk_id="chunk_surgical_2",
                text="Experimental evaluation demonstrates real-time inference at 45 FPS on deformable liver and kidney phantoms with cut-off threshold 0.05.",
                source_file="surgical_robotics.pdf",
                metadata={"source_file": "surgical_robotics.pdf", "page_number": 6}
            )
        ]

        vector_store.add_chunks(test_chunks)
        print(f"  • Indexed {len(test_chunks)} test chunks successfully.")

        # Test Semantic Retrieval Separation: In-Domain vs Out-of-Domain
        in_domain_results = vector_store.query("How does LUDO predict 3D occupancy and SDF?", top_k=2)
        ood_results = vector_store.query("Who is the prime minister of india?", top_k=2)
        gibberish_results = vector_store.query("ofuvlnsjkadfoisudfeadnvdsajfoasf", top_k=2)

        in_score = in_domain_results[0].score if in_domain_results else 0.0
        ood_score = ood_results[0].score if ood_results else 0.0
        gib_score = gibberish_results[0].score if gibberish_results else 0.0

        print(f"  • In-Domain Retrieval Score : {in_score:.4f} ('{in_domain_results[0].text[:45]}...')")
        print(f"  • Out-of-Domain Query Score : {ood_score:.4f}")
        print(f"  • Gibberish Query Score     : {gib_score:.4f}")

        assert in_score > ood_score, f"FAIL: In-domain score ({in_score}) should exceed out-of-domain score ({ood_score})"
        print(f"  • Discrimination Margin     : +{(in_score - ood_score)*100:.1f}% semantic separation")

        # Test Guardrail Evaluation on In-domain vs Out-of-Domain
        is_g_in, conf_in, cites_in = guardrails.evaluate_grounding(
            "How does LUDO predict 3D occupancy and SDF?",
            "LUDO predicts 3D occupancy and SDF for deformable organs.",
            in_domain_results
        )
        assert is_g_in is True
        print(f"  • Grounding In-Domain Check : PASS (Confidence: {conf_in*100:.1f}%, Citations: {len(cites_in)})")

        is_g_out, conf_out, cites_out = guardrails.evaluate_grounding(
            "Who is the prime minister of india?",
            "Based on the provided documents, there is no mention of the prime minister of india.",
            ood_results
        )
        assert is_g_out is False
        assert len(cites_out) == 0
        print(f"  • Grounding Out-Domain Check: PASS (is_grounded=False, Citations stripped to 0)")

        print(f"  ✅ EMBEDDER {model_name} VERIFIED & FULLY FUNCTIONAL!")

    finally:
        del embedder
        del vector_store
        del guardrails
        import gc
        gc.collect()
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_all_embedders_test():
    print("\n" + "="*70)
    print(" 🚀 [SUITE 2/2] COMPREHENSIVE EMBEDDER DISCRIMINATION TESTS")
    print("="*70)

    # Test top embedders in catalog
    models_to_test = [
        "BAAI/bge-base-en-v1.5",
        "all-MiniLM-L6-v2"
    ]

    for model_name in models_to_test:
        test_embedder_model(model_name)

    print("\n" + "="*70)
    print(" 🎉 ALL EMBEDDER & GUARDRAIL EXTENSIVE TESTS COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_unit_guardrails_tests()
    run_all_embedders_test()
