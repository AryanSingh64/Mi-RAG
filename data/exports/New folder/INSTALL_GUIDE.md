# 🚀 Self-Hosted Production RAG Deployment Guide

This package contains your fully indexed, standalone enterprise RAG pipeline.

---

## ⚡ Option 1: One-Click Quick Start (Windows)
1. Make sure **Ollama** is running on your machine:
   ```bash
   ollama pull llama3.2:1b
   ```
2. Double-click **`run.bat`** (or run `.\run.bat` in PowerShell).
3. Open your browser to: **`http://localhost:8000`**

---

## 🐧 Option 2: Linux / macOS Startup
1. Make executable and run:
   ```bash
   chmod +x run.sh
   ./run.sh
   ```
2. Open **`http://localhost:8000`**

---

## 🐳 Option 3: Docker Deployment
```bash
docker compose up --build
```
