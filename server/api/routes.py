import json
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
import httpx
from pydantic import BaseModel

from core.llm.model_hub import ModelHub
from core.llm.ollama_client import OllamaClient
from core.system.hardware_detector import HardwareDetector
from packager.exporter import RAGPackager
from server.sessions.session_manager import SessionManager

router = APIRouter()
session_manager = SessionManager(base_storage_dir="./data/sessions", default_ttl_hours=3.0)
packager = RAGPackager(export_output_dir="./data/exports")
ollama_client = OllamaClient()


# Request/Response Schemas
class CreateSessionRequest(BaseModel):
    model_name: Optional[str] = "llama3.2:3b"
    vision_models: Optional[List[str]] = ["moondream"]
    vision_model: Optional[str] = None  # Backward compatibility
    embedding_model: Optional[str] = "all-MiniLM-L6-v2"
    ttl_hours: Optional[float] = 3.0


class ChatRequest(BaseModel):
    message: str
    top_k: Optional[int] = 6


@router.get("/system/specs")
def get_system_specs():
    """Returns detected hardware specifications (GPU, VRAM, RAM, CPU threads)."""
    return HardwareDetector.get_specs()


@router.get("/models/hub")
def get_models_hub():
    """Returns full catalog of trending & local models with hardware compatibility ratings."""
    return ModelHub.get_catalog_with_status()


@router.get("/models/pull/stream")
async def pull_model_stream(model: str):
    """
    Streams live download progress for pulling an Ollama model globally to the user's PC.
    """
    target_model = model.strip()

    async def event_generator():
        # Immediate connection ping
        yield f"data: {json.dumps({'status': f'Connecting to Ollama to pull {target_model}...', 'completed': 0, 'total': 100})}\n\n"
        payload = {"name": target_model, "stream": True}
        try:
            async with httpx.AsyncClient(timeout=1800.0) as client:
                async with client.stream("POST", "http://localhost:11434/api/pull", json=payload) as response:
                    if response.status_code != 200:
                        yield f"data: {json.dumps({'status': f'Ollama error status {response.status_code}'})}\n\n"
                        return
                    async for line in response.aiter_lines():
                        if line and line.strip():
                            yield f"data: {line.strip()}\n\n"
        except Exception as e:
            err_data = json.dumps({"status": "error", "error": str(e)})
            yield f"data: {err_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream"
        }
    )


@router.get("/models")
def get_available_models():
    """
    Returns local Ollama models classified into Text Models and Vision Models.
    """
    all_models = ollama_client.list_local_models()
    vision_keywords = ["vision", "moondream", "llava", "minicpm", "bakllava", "cogvlm", "vl", "ocr"]

    text_models = []
    vision_models = []

    for m in all_models:
        m_lower = m.lower()
        if any(vk in m_lower for vk in vision_keywords):
            vision_models.append(m)
        else:
            text_models.append(m)

    # Defaults if not installed
    if not text_models:
        text_models = ["llama3.2:3b", "llama3.2:1b"]
    if not vision_models:
        vision_models = ["moondream:latest"]

    return {
        "text_models": text_models,
        "vision_models": vision_models,
        "models": all_models
    }


@router.post("/sessions/create")
def create_session(req: CreateSessionRequest):
    """Creates a new ephemeral RAG session with optional multi-model Vision ensemble."""
    models_to_use = req.vision_models
    if not models_to_use and req.vision_model:
        models_to_use = [req.vision_model]
    if not models_to_use:
        models_to_use = ["moondream"]

    session = session_manager.create_session(
        model_name=req.model_name or "llama3.2:3b",
        vision_models=models_to_use,
        embedding_model=req.embedding_model or "all-MiniLM-L6-v2",
        ttl_hours=req.ttl_hours or 3.0
    )
    return {
        "session_id": session.session_id,
        "model_name": session.model_name,
        "expires_at": session.expires_at,
        "time_remaining_seconds": session.time_remaining_seconds,
        "portal_url": f"/portal/{session.session_id}"
    }


@router.get("/sessions/{session_id}")
def get_session_info(session_id: str):
    """Returns status, time remaining, and indexed files for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired or not found.")
    return {
        "session_id": session.session_id,
        "model_name": session.model_name,
        "indexed_files": session.indexed_files,
        "time_remaining_seconds": session.time_remaining_seconds,
        "is_expired": session.is_expired
    }


@router.post("/sessions/{session_id}/upload")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    """Uploads and ingests any supported document (PDF, DOCX, TXT, Image/OCR)."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired or not found.")

    file_path = session.uploads_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunks_indexed = session.pipeline.ingest_file(file_path)
        session.indexed_files.append(file.filename)
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_indexed": chunks_indexed,
            "total_files": len(session.indexed_files)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.post("/sessions/{session_id}/chat")
def chat_with_rag(session_id: str, req: ChatRequest):
    """Executes a grounded query against the session's private knowledge base."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired or not found.")

    answer = session.pipeline.query(req.message, top_k=req.top_k)
    return {
        "answer": answer.answer,
        "confidence_score": answer.confidence_score,
        "is_grounded": answer.is_grounded,
        "citations": [
            {
                "source_file": c.source_file,
                "relevance": round(c.score * 100, 1),
                "text": c.text[:200] + "..." if len(c.text) > 200 else c.text
            }
            for c in answer.citations
        ]
    }


@router.get("/sessions/{session_id}/export")
def export_package(session_id: str):
    """Creates and downloads the standalone ZIP package."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired or not found.")

    zip_path = packager.create_package(session)
    return FileResponse(
        path=str(zip_path),
        filename=f"rag_deployment_{session.session_id}.zip",
        media_type="application/zip"
    )
