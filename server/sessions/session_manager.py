import asyncio
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from core.pipeline import RAGPipeline


@dataclass
class RAGSession:
    """
    Represents an isolated, time-limited RAG deployment session.
    """
    session_id: str
    created_at: float
    expires_at: float
    session_dir: Path
    db_dir: Path
    uploads_dir: Path
    images_dir: Path
    pipeline: RAGPipeline
    model_name: str
    indexed_files: List[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def time_remaining_seconds(self) -> int:
        return max(0, int(self.expires_at - time.time()))


class SessionManager:
    """
    Manages multi-tenant ephemeral RAG sessions with automatic TTL expiration.
    """

    def __init__(self, base_storage_dir: Path | str = "./data/sessions", default_ttl_hours: float = 3.0):
        self.base_dir = Path(base_storage_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl_seconds = int(default_ttl_hours * 3600)
        self.active_sessions: Dict[str, RAGSession] = {}

    def create_session(
        self,
        model_name: str = "llama3.2:3b",
        vision_models: Optional[List[str] | str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        ttl_hours: Optional[float] = None
    ) -> RAGSession:
        """
        Creates a new isolated session workspace with private vector store and extracted image gallery.
        """
        session_id = uuid.uuid4().hex[:12]  # Short 12-char secure token
        now = time.time()
        ttl = int((ttl_hours or 3.0) * 3600)

        session_dir = self.base_dir / session_id
        db_dir = session_dir / "vector_db"
        uploads_dir = session_dir / "uploads"
        images_dir = session_dir / "extracted_images"

        db_dir.mkdir(parents=True, exist_ok=True)
        uploads_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

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
            created_at=now,
            expires_at=now + ttl,
            session_dir=session_dir,
            db_dir=db_dir,
            uploads_dir=uploads_dir,
            images_dir=images_dir,
            pipeline=pipeline,
            model_name=model_name
        )

        self.active_sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[RAGSession]:
        """
        Retrieves an active session if not expired.
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return None

        if session.is_expired:
            self.delete_session(session_id)
            return None

        return session

    def delete_session(self, session_id: str):
        """
        Removes session and permanently wipes private data from disk.
        """
        session = self.active_sessions.pop(session_id, None)
        if session and session.session_dir.exists():
            try:
                shutil.rmtree(session.session_dir)
            except Exception as e:
                print(f"[!] Error deleting session {session_id}: {e}")

    def cleanup_expired_sessions(self):
        """
        Background cleaner to wipe expired sessions.
        """
        expired = [sid for sid, s in self.active_sessions.items() if s.is_expired]
        for sid in expired:
            self.delete_session(sid)
