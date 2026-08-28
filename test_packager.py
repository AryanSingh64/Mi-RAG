import sys
from pathlib import Path

# Enable UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from server.sessions.session_manager import SessionManager
from packager.exporter import RAGPackager

def test_packager():
    print("==================================================")
    print("[*] Initializing SessionManager & Standalone Packager...")
    print("==================================================")
    
    mgr = SessionManager(base_storage_dir="./data/sessions", default_ttl_hours=2.0)
    packager = RAGPackager(export_output_dir="./data/exports")

    # 1. Create an ephemeral session
    session = mgr.create_session(model_name="llama3.2:1b")
    print(f"[+] Created Ephemeral Session Token: {session.session_id}")
    print(f"[+] Expires in: {session.time_remaining_seconds} seconds ({session.time_remaining_seconds/3600:.1f} hours)")

    # 2. Ingest a document into this session
    doc_path = session.uploads_dir / "client_policy.txt"
    doc_path.write_text(
        "Confidential Project Titan Specification:\n"
        "Titan is an autonomous edge computing device with 512GB NVMe storage and dual 10GbE network interfaces.",
        encoding="utf-8"
    )
    chunks = session.pipeline.ingest_file(doc_path)
    session.indexed_files.append("client_policy.txt")
    print(f"[+] Ingested {chunks} chunks into session private ChromaDB.")

    # 3. Export Standalone Package (ZIP)
    zip_path = packager.create_package(session)
    print(f"\n[+] Standalone ZIP Created: {zip_path}")
    print(f"[+] ZIP Size: {zip_path.stat().st_size / 1024:.1f} KB")

    assert zip_path.exists(), "ZIP package should exist"
    print("\n==================================================")
    print("[SUCCESS] Session manager & Standalone Packager test passed!")
    print("==================================================")

if __name__ == "__main__":
    test_packager()
