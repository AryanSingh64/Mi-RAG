import json
import shutil
import uuid
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
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


from core.llm.multi_provider import MultiProviderLLM


class KeyTestRequest(BaseModel):
    provider: str
    api_key: Optional[str] = None
    model: Optional[str] = None


@router.get("/providers")
def get_providers():
    """Returns available model providers, presets, and configuration options."""
    return MultiProviderLLM.get_provider_presets()


@router.post("/keys/test")
def test_provider_key(req: KeyTestRequest):
    """Tests validity of an API key against the provider's endpoint."""
    return MultiProviderLLM.test_key(provider=req.provider, api_key=req.api_key or "", model=req.model)


@router.post("/models/fetch")
def fetch_provider_models(req: KeyTestRequest):
    """Fetches latest dynamic model catalogue from the provider's API."""
    models = MultiProviderLLM.fetch_models(provider=req.provider, api_key=req.api_key or "")
    return {"provider": req.provider, "models": models}


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


@router.delete("/models")
async def delete_model(model: str):
    """
    Deletes an installed model from local Ollama to free up disk space and VRAM.
    """
    target_model = model.strip()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.request("DELETE", "http://localhost:11434/api/delete", json={"name": target_model})
            if res.status_code == 200:
                return {"status": "success", "message": f"Deleted model {target_model}"}
            else:
                return {"status": "error", "message": res.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/embeddings")
def get_embedding_models():
    """Returns available modular embedding models and configurations."""
    from core.embeddings.embedder import LocalEmbedder
    return {"catalog": LocalEmbedder.get_catalog(), "default": "BAAI/bge-base-en-v1.5"}


@router.post("/sessions/create")
def create_session(req: CreateSessionRequest):
    """Creates a new ephemeral RAG session with modular embedding & multi-model Vision ensemble."""
    models_to_use = req.vision_models
    if not models_to_use and req.vision_model:
        models_to_use = [req.vision_model]
    if not models_to_use:
        models_to_use = ["moondream"]

    chosen_embedder = req.embedding_model or "BAAI/bge-base-en-v1.5"

    session = session_manager.create_session(
        model_name=req.model_name or "llama3.2:3b",
        vision_models=models_to_use,
        embedding_model=chosen_embedder,
        ttl_hours=req.ttl_hours or 3.0
    )
    print(f"\n[SESSION CREATED] ID: {session.session_id} | Model: {session.model_name} | Vision: {models_to_use} | Embedder: {session.embedding_model}")
    return {
        "session_id": session.session_id,
        "model_name": session.model_name,
        "embedding_model": session.embedding_model,
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
    import time
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired or not found.")

    file_path = session.uploads_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"\n[INGESTION START] Session: {session_id} | File: {file.filename} ({file_size_mb:.2f} MB)")
    t0 = time.perf_counter()

    try:
        chunks_indexed = session.pipeline.ingest_file(file_path)
        dur = time.perf_counter() - t0
        session.indexed_files.append(file.filename)
        print(f"[INGESTION COMPLETE] {file.filename} -> {chunks_indexed} vectors indexed in {dur:.2f}s ({file_size_mb / max(dur, 0.001):.1f} MB/s)")
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_indexed": chunks_indexed,
            "elapsed_seconds": round(dur, 2),
            "total_files": len(session.indexed_files)
        }
    except Exception as e:
        print(f"[INGESTION ERROR] {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.post("/sessions/{session_id}/chat")
async def chat_with_rag(session_id: str, request: Request):
    """
    Executes a grounded query against the session's private knowledge base.
    Supports both text queries (JSON) and Multimodal Visual Searches with image attachments (Multipart Form).
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired or not found.")

    content_type = request.headers.get("content-type", "")
    query_image_path = None
    query_image_url = None
    user_message = ""
    history = None
    top_k = 6
    provider = "ollama"
    model = None
    api_key = None

    if "application/json" in content_type:
        try:
            data = await request.json()
            user_message = data.get("message", "")
            top_k = int(data.get("top_k", 6))
            history = data.get("history", None)
            provider = str(data.get("provider", "ollama")).strip()
            model = data.get("model", None)
            api_key = data.get("api_key", None)
        except Exception:
            pass
    else:
        form = await request.form()
        user_message = str(form.get("message", "")).strip()
        provider = str(form.get("provider", "ollama")).strip()
        model = form.get("model", None)
        api_key = form.get("api_key", None)
        try:
            top_k = int(form.get("top_k", 6))
        except Exception:
            top_k = 6
        
        hist_raw = form.get("history")
        if hist_raw:
            try:
                import json
                history = json.loads(hist_raw)
            except Exception:
                pass

        uploaded_image = form.get("image")
        if uploaded_image and hasattr(uploaded_image, "filename") and uploaded_image.filename:
            clean_fname = f"query_{uuid.uuid4().hex[:8]}_{uploaded_image.filename}"
            query_image_path = session.uploads_dir / clean_fname
            with open(query_image_path, "wb") as buffer:
                shutil.copyfileobj(uploaded_image.file, buffer)
            query_image_url = f"/api/sessions/{session_id}/images/{clean_fname}"

    try:
        if query_image_path and query_image_path.exists():
            answer = session.pipeline.query_with_image(
                user_question=user_message,
                query_image_path=query_image_path,
                top_k=top_k,
                history=history,
                provider=provider,
                model=model,
                api_key=api_key
            )
        else:
            answer = session.pipeline.query(
                user_question=user_message,
                top_k=top_k,
                history=history,
                provider=provider,
                model=model,
                api_key=api_key
            )
    except Exception as e:
        print(f"[!] Chat processing error: {e}")
        return {
            "answer": f"Unable to complete visual analysis or query: {str(e)}",
            "confidence_score": 0.0,
            "is_grounded": False,
            "citations": [],
            "images": [],
            "query_image_url": query_image_url,
            "error": str(e)
        }

    # Deduplicate citations by source file and keep highest similarity score
    unique_citations = {}
    for c in answer.citations:
        score_val = float(c.score) if c.score is not None else 0.0
        score_pct = round(score_val * 100, 1)
        if c.source_file not in unique_citations or score_pct > unique_citations[c.source_file]["relevance"]:
            unique_citations[c.source_file] = {
                "source_file": c.source_file,
                "relevance": score_pct,
                "text": c.text[:200] + "..." if len(c.text) > 200 else c.text
            }

    return {
        "answer": answer.answer,
        "confidence_score": answer.confidence_score,
        "is_grounded": answer.is_grounded,
        "citations": list(unique_citations.values()),
        "images": answer.images,
        "query_image_url": query_image_url
    }


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: str  # "thumbs_up" | "thumbs_down"
    model: Optional[str] = None
    citations: Optional[List[Any]] = None


@router.post("/sessions/{session_id}/feedback")
def submit_feedback(session_id: str, req: FeedbackRequest):
    """Records user feedback (thumbs_up / thumbs_down) for continuous in-context learning and DPO dataset export."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired or not found.")
    
    entry = session.pipeline.record_feedback(
        query=req.query,
        answer=req.answer,
        rating=req.rating,
        model=req.model,
        citations=req.citations
    )
    return {
        "status": "success",
        "message": f"Feedback recorded as {req.rating}",
        "total_feedback": len(session.pipeline.feedback_store)
    }


@router.get("/sessions/{session_id}/feedback/export")
def export_feedback(session_id: str):
    """Exports all session feedback interactions formatted as a DPO / SFT training dataset."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired or not found.")
    return {
        "session_id": session_id,
        "dataset": session.pipeline.feedback_store,
        "total_items": len(session.pipeline.feedback_store)
    }


@router.post("/sessions/{session_id}/clear_memory")
def clear_session_memory(session_id: str):
    """Resets the multi-turn conversational memory for a session."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired or not found.")
    session.pipeline.clear_memory()
    return {"status": "success", "message": "Memory cleared."}


@router.get("/sessions/{session_id}/images/{filename}")
def get_session_image(session_id: str, filename: str):
    """Serves an extracted diagram or image snippet from the session knowledge base."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired or not found.")

    image_path = session.images_dir / filename
    if not image_path.exists():
        # Check in uploads directory as fallback
        image_path = session.uploads_dir / filename

    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image {filename} not found.")

    # Determine media type
    suffix = image_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif"
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(
        path=str(image_path),
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=3600"}
    )


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
