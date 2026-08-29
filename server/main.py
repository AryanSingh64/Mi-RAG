import os
import sys
from pathlib import Path

# Add project root directory to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Windows UTF-8 stdout fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

from fastapi.staticfiles import StaticFiles
from server.api.routes import router as api_router

app = FastAPI(
    title="Mi:RAG Autonomous Factory",
    description="Zero-budget, local-first Multimodal RAG generator and deployment exporter",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets (images, logos, backgrounds)
static_dir = ROOT_DIR / "web" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include API routes
app.include_router(api_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
@app.get("/studio", response_class=HTMLResponse)
def app_studio():
    html_path = ROOT_DIR / "web" / "templates" / "app.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Mi:RAG Studio is loading...</h1>"


@app.get("/docs", response_class=HTMLResponse)
@app.get("/docs.html", response_class=HTMLResponse)
def docs_page():
    html_path = ROOT_DIR / "web" / "templates" / "docs.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Mi:RAG Documentation</h1>"


@app.get("/portal/{session_id}", response_class=HTMLResponse)
def portal(session_id: str):
    html_path = ROOT_DIR / "web" / "templates" / "portal.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return f"<h1>Ephemeral Portal for Session: {session_id}</h1>"


if __name__ == "__main__":
    print("==================================================")
    print("🚀 Starting Autonomous RAG Factory Web App...")
    print("🌐 Access Dashboard at: http://localhost:8000")
    print("==================================================")
    uvicorn.run(app, host="127.0.0.1", port=8000)

