@echo off
title Autonomous RAG Factory
echo ==================================================
echo Starting Autonomous RAG Factory...
echo ==================================================
call .venv\Scripts\activate.bat
python run_factory.py
pause
