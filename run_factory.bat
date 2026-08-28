@echo off
title Mi:RAG - Autonomous Multimodal RAG Engine
echo ==================================================
echo Starting Mi:RAG Engine...
echo ==================================================

if not exist .venv (
    echo [*] Creating virtual environment (.venv)...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

python -c "import uvicorn, fastapi" 2>nul
if %errorlevel% neq 0 (
    echo [*] Installing required packages...
    pip install --upgrade pip
    pip install -r requirements.txt
)

python run_factory.py
pause
