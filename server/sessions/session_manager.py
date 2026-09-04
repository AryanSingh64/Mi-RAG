import asyncio
import hmac
import json
import os
import secrets
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.pipeline import RAGPipeline


@dataclass
class RAGSession:
    """
    Represents an isolated, time-limited RAG deployment session with strict ACID ownership.
    """
    session_id: str
    session_token: str
    created_at: float
    expires_at: float
    session_dir: Path
    db_dir: Path
    uploads_dir: Path
    images_dir: Path
    pipeline: RAGPipeline
    model_name: str
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    indexed_files: List[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def time_remaining_seconds(self) -> int:
        return max(0, int(self.expires_at - time.time()))


class SessionManager:
    """
    Manages multi-tenant ephemeral RAG sessions with ACID properties:
    - Atomicity: Workspace and pipeline initialize cleanly or rollback completely on failure.
    - Consistency: 1-to-1 cryptographic session tokens (preventing timing attacks).
    - Isolation: Granular per-session locks prevent concurrent race conditions without blocking other sessions.
    - Durability: Atomic fsync metadata persistence allows transparent recovery after restarts.
    """

    def __init__(self, base_storage_dir: Path | str = "./data/sessions", default_ttl_hours: float = 3.0):
        self.base_dir = Path(base_storage_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl_seconds = int(default_ttl_hours * 3600)
        self.active_sessions: Dict[str, RAGSession] = {}
        self._lock = threading.RLock()
        self._progress_store: Dict[str, Dict[str, Any]] = {}

    def _atomic_write_meta(self, session_dir: Path, data: dict):
        """Durability: Atomic fsync write prevents metadata corruption on sudden power loss/crashes."""
        meta_file = session_dir / "meta.json"
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(session_dir), prefix="meta_tmp_", suffix=".json")
        try:
            with open(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, meta_file)
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            raise e

    def create_session(
        self,
        model_name: str = "llama3.2:3b",
        vision_models: Optional[List[str] | str] = None,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        ttl_hours: Optional[float] = None
    ) -> RAGSession:
        """
        Atomically creates a new isolated session workspace with private vector store and extracted image gallery.
        Rolls back all disk changes if initialization fails.
        """
        session_id = uuid.uuid4().hex[:12]
        session_token = secrets.token_urlsafe(32)  # High-entropy cryptographic token
        now = time.time()
        ttl = int((ttl_hours or 3.0) * 3600)

        session_dir = self.base_dir / session_id
        db_dir = session_dir / "vector_db"
        uploads_dir = session_dir / "uploads"
        images_dir = session_dir / "extracted_images"

        # ATOMICITY: Setup workspace or rollback cleanly on any error
        try:
            db_dir.mkdir(parents=True, exist_ok=True)
            uploads_dir.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)

            meta_data = {
                "session_id": session_id,
                "session_token": session_token,
                "created_at": now,
                "expires_at": now + ttl,
                "model_name": model_name,
                "embedding_model": embedding_model,
                "vision_models": vision_models or ["moondream"],
                "indexed_files": []
            }
            self._atomic_write_meta(session_dir, meta_data)

            pipeline = RAGPipeline(
                persist_directory=db_dir,
                collection_name=f"coll_{session_id}",
                embedding_model=embedding_model,
                ollama_model=model_name,
                vision_models=vision_models or ["moondream"],
                extracted_images_dir=images_dir,
                session_id=session_id
            )

            session = RAGSession(
                session_id=session_id,
                session_token=session_token,
                created_at=now,
                expires_at=now + ttl,
                session_dir=session_dir,
                db_dir=db_dir,
                uploads_dir=uploads_dir,
                images_dir=images_dir,
                pipeline=pipeline,
                model_name=model_name,
                embedding_model=embedding_model,
                indexed_files=[]
            )

            with self._lock:
                self.active_sessions[session_id] = session

            return session

        except Exception as e:
            # Atomic rollback
            if session_dir.exists():
                shutil.rmtree(session_dir, ignore_errors=True)
            print(f"[!] Session creation rollback: {e}")
            raise e

    def validate_session_token(self, session_id: str, token: str) -> bool:
        """
        CONSISTENCY: Constant-time authentication preventing timing attacks and unauthorized access.
        """
        if not session_id or not token:
            return False

        with self._lock:
            session = self.get_session(session_id)
            if not session or session.is_expired:
                return False
            return hmac.compare_digest(session.session_token, token)

    def get_session(self, session_id: str) -> Optional[RAGSession]:
        """
        Retrieves an active session, automatically rehydrating from disk if server was restarted (Durability).
        """
        with self._lock:
            session = self.active_sessions.get(session_id)

            if not session:
                session_dir = self.base_dir / session_id
                meta_file = session_dir / "meta.json"
                if session_dir.exists() and meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)

                        # Check expiration
                        if time.time() > meta.get("expires_at", 0):
                            self.delete_session(session_id)
                            return None

                        db_dir = session_dir / "vector_db"
                        uploads_dir = session_dir / "uploads"
                        images_dir = session_dir / "extracted_images"

                        pipeline = RAGPipeline(
                            persist_directory=db_dir,
                            collection_name=f"coll_{session_id}",
                            embedding_model=meta.get("embedding_model", "BAAI/bge-base-en-v1.5"),
                            ollama_model=meta.get("model_name", "llama3.2:3b"),
                            vision_models=meta.get("vision_models", ["moondream"]),
                            extracted_images_dir=images_dir,
                            session_id=session_id
                        )

                        session = RAGSession(
                            session_id=session_id,
                            session_token=meta.get("session_token", ""),
                            created_at=meta.get("created_at", time.time()),
                            expires_at=meta.get("expires_at", time.time() + 10800),
                            session_dir=session_dir,
                            db_dir=db_dir,
                            uploads_dir=uploads_dir,
                            images_dir=images_dir,
                            pipeline=pipeline,
                            model_name=meta.get("model_name", "llama3.2:3b"),
                            embedding_model=meta.get("embedding_model", "BAAI/bge-base-en-v1.5"),
                            indexed_files=meta.get("indexed_files", [])
                        )
                        self.active_sessions[session_id] = session
                    except Exception as e:
                        print(f"[!] Error rehydrating session {session_id} from disk: {e}")
                        return None

            if not session:
                return None

            if session.is_expired:
                self.delete_session(session_id)
                return None

            return session

    def update_session_indexed_files(self, session_id: str, new_files: List[str]):
        """
        Atomically updates indexed files list in memory and on disk under session lock.
        """
        session = self.get_session(session_id)
        if session:
            with session.lock:
                for f in new_files:
                    if f not in session.indexed_files:
                        session.indexed_files.append(f)

                meta_file = session.session_dir / "meta.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as mf:
                            meta = json.load(mf)
                        meta["indexed_files"] = session.indexed_files
                        self._atomic_write_meta(session.session_dir, meta)
                    except Exception as e:
                        print(f"[!] Error persisting updated indexed files: {e}")

    def delete_session(self, session_id: str):
        """
        Atomically wipes session, ChromaDB collection, and workspace from disk.
        """
        with self._lock:
            session = self.active_sessions.pop(session_id, None)
            self._progress_store.pop(session_id, None)

        session_dir = self.base_dir / session_id
        if session_dir.exists():
            try:
                shutil.rmtree(session_dir, ignore_errors=True)
            except Exception as e:
                print(f"[!] Error deleting session directory {session_id}: {e}")

    def set_progress(self, session_id: str, stage: str, current: int, total: int, diagrams: int = 0, status: str = "processing"):
        """Thread-safe updates to live ingestion progress."""
        pct = round((current / max(1, total)) * 100, 1) if total > 0 else 0.0
        with self._lock:
            self._progress_store[session_id] = {
                "status": status,
                "stage": stage,
                "current": current,
                "total": total,
                "percent": min(100.0, pct),
                "diagrams": diagrams,
                "updated_at": time.time()
            }

    def get_progress(self, session_id: str) -> Dict[str, Any]:
        """Thread-safe read of ingestion progress."""
        with self._lock:
            return self._progress_store.get(session_id, {
                "status": "idle",
                "stage": "Ready",
                "current": 0,
                "total": 0,
                "percent": 0.0,
                "diagrams": 0
            })
