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
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

from fastapi.staticfiles import StaticFiles
from server.api.routes import router as api_router, session_manager

app = FastAPI(
    title="Mi:RAG Autonomous Factory",
    description="Zero-budget, local-first Multimodal RAG generator and deployment exporter",
    version="1.0.0"
)

# Custom ACID-compliant Session Authentication Middleware
class SessionAuthMiddleware(BaseHTTPMiddleware):
    """
    Guards all sensitive session APIs (/api/sessions/{id}/*) requiring
    cryptographically verified session ownership tokens.
    """
    async def dispatch(self, request, call_next):
        path = request.url.path

        # 1. Whitelist all public routes, static files, and hub catalog
        if (
            path in ["/", "/app", "/docs", "/docs.html", "/studio", "/favicon.ico"]
            or path.startswith("/static")
            or path.startswith("/api/system")
            or path.startswith("/api/models")
            or path.startswith("/api/providers")
            or path.startswith("/api/embeddings")
            or path.startswith("/api/keys")
            or path == "/api/documents/inspect"
            or (path == "/api/sessions/create" and request.method == "POST")
            or path.startswith("/portal/")
        ):
            return await call_next(request)

        # 2. Intercept and guard all session resources: /api/sessions/{session_id}/*
        if path.startswith("/api/sessions/"):
            parts = path.split("/")
            if len(parts) >= 4 and parts[3]:
                session_id = parts[3]

                # Extract token from Header (X-Session-Token or Authorization: Bearer) or query param
                session_token = request.headers.get("X-Session-Token", "")
                if not session_token:
                    auth_header = request.headers.get("Authorization", "")
                    if auth_header.startswith("Bearer "):
                        session_token = auth_header[7:].strip()
                if not session_token:
                    session_token = request.query_params.get("token", "")

                if not session_token:
                    return JSONResponse(
                        status_code=401,
                        content={"error": "Unauthorized: Missing session authentication token. Access denied."}
                    )

                if not session_manager.validate_session_token(session_id, session_token):
                    return JSONResponse(
                        status_code=403,
                        content={"error": "Forbidden: Invalid or expired session authentication token."}
                    )

        return await call_next(request)

app.add_middleware(SessionAuthMiddleware)
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


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    fav_path = ROOT_DIR / "web" / "static" / "assets" / "favicon.ico"
    if fav_path.exists():
        return FileResponse(fav_path)
    return HTMLResponse(status_code=404)


if __name__ == "__main__":
    print("==================================================")
    print("🚀 Starting Autonomous RAG Factory Web App...")
    print("🌐 Access Dashboard at: http://localhost:8000")
    print("==================================================")
    uvicorn.run(app, host="127.0.0.1", port=8000)

