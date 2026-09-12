"""
Local SQLite-backed Conversation & Chat History Repository
Provides ACID-compliant, persistent multi-chat storage that persists across browser refreshes
and travels with standalone exported packages.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


class ChatStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    @contextlib.contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        try:
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _init_db(self) -> None:
        with self._lock, self._connection() as conn:
            conn.executescript("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        image_url TEXT,
                        grounding_metadata TEXT,
                        created_at REAL NOT NULL,
                        FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
                    CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
                    CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at ASC);
                """)
            conn.commit()

    def create_conversation(self, session_id: str, title: str = "New Chat") -> Dict[str, Any]:
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        now = time.time()
        with self._lock, self._connection() as conn:
            conn.execute(
                "INSERT INTO conversations (id, session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (conv_id, session_id, title.strip() or "New Chat", now, now)
            )
            conn.commit()
        return {
            "id": conv_id,
            "session_id": session_id,
            "title": title.strip() or "New Chat",
            "created_at": now,
            "updated_at": now,
            "message_count": 0
        }

    def list_conversations(self, session_id: str) -> List[Dict[str, Any]]:
        with self._lock, self._connection() as conn:
            rows = conn.execute("""
                SELECT c.id, c.session_id, c.title, c.created_at, c.updated_at,
                       COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                WHERE c.session_id = ?
                GROUP BY c.id
                ORDER BY c.updated_at DESC
            """, (session_id,)).fetchall()
            return [dict(r) for r in rows]

    def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connection() as conn:
            c_row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
            if not c_row:
                return None
            conv = dict(c_row)
            m_rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conv_id,)
            ).fetchall()
            messages = []
            for mr in m_rows:
                m = dict(mr)
                if m.get("grounding_metadata"):
                    try:
                        m["grounding_metadata"] = json.loads(m["grounding_metadata"])
                    except Exception:
                        pass
                messages.append(m)
            conv["messages"] = messages
            conv["message_count"] = len(messages)
            return conv

    def update_conversation_title(self, conv_id: str, title: str) -> bool:
        clean_title = title.strip()[:60]
        if not clean_title:
            clean_title = "New Chat"
        now = time.time()
        with self._lock, self._connection() as conn:
            res = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (clean_title, now, conv_id)
            )
            conn.commit()
            return res.rowcount > 0

    def touch_conversation(self, conv_id: str) -> None:
        now = time.time()
        with self._lock, self._connection() as conn:
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
            conn.commit()

    def delete_conversation(self, conv_id: str) -> bool:
        with self._lock, self._connection() as conn:
            res = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            conn.commit()
            return res.rowcount > 0

    def add_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        image_url: Optional[str] = None,
        grounding_metadata: Optional[Any] = None
    ) -> Dict[str, Any]:
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"
        now = time.time()
        meta_json = json.dumps(grounding_metadata) if grounding_metadata else None

        with self._lock, self._connection() as conn:
            conn.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, image_url, grounding_metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (msg_id, conv_id, role, content, image_url, meta_json, now)
            )
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
            conn.commit()

        return {
            "id": msg_id,
            "conversation_id": conv_id,
            "role": role,
            "content": content,
            "image_url": image_url,
            "grounding_metadata": grounding_metadata,
            "created_at": now
        }

    def generate_smart_title(self, first_user_message: str) -> str:
        """Derives a clean ChatGPT-like conversation title from the first message."""
        text = first_user_message.strip().replace("\r", " ").replace("\n", " ")
        # Strip common leading question phrases if any
        low = text.lower()
        for prefix in ["what is ", "what are ", "can you explain ", "tell me about ", "show me "]:
            if low.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        if len(text) > 48:
            text = text[:45].rstrip() + "..."
        # Capitalize first letter
        return text[0].upper() + text[1:] if text else "New Chat"
