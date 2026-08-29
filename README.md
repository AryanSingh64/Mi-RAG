# Mi:RAG

**Autonomous, zero-budget, 100% offline Multimodal RAG Engine & Turnkey Package Exporter.**

Mi:RAG indexes your documents (PDFs, Word documents, images) on your local machine, extracts text, tables, formulas, and visual diagrams, and exports a standalone offline assistant bundle that runs anywhere with zero cloud dependencies or API keys.

---

## Key Features

- **1-Line Installation**: Automated hardware profiling, Ollama detection, and embedding setup via PowerShell or Git.
- **Multimodal Visual Parser**: Uses vector clustering (PyMuPDF) and OCR to detect, crop, and index diagrams, figures, and charts alongside text.
- **Reverse Visual Search**: Paste (Ctrl+V) or upload an image to search for matching diagrams, formulas, and architectural drawings across documents.
- **Multi-Turn Memory & Attention**: Context retention across follow-up queries with client-side persistence and reset controls.
- **Anti-Hallucination Guardrails**: Cross-references every LLM claim against retrieved source chunks with confidence scoring.
- **Turnkey Standalone Export**: Generates a self-contained ZIP bundle with pre-indexed ChromaDB, FastAPI microservice, and 1-click execution scripts (`run.bat` / `run.sh`).
- **Zero Cloud Leakage**: All inference, vector storage, and processing execute on-premise on your local hardware.

---

## Quickstart & Installation

### Option 1: Windows 1-Line Installer (PowerShell)

Run PowerShell as Administrator and execute:

```powershell
irm https://mirag.me/install | iex
```

This automated script:
1. Detects your hardware (NVIDIA GPU / CPU cores / RAM).
2. Verifies your Ollama installation and pulls the recommended model (e.g. `llama3.2:3b` or `qwen2.5:3b`).
3. Caches local sentence-transformer embedding weights (`all-MiniLM-L6-v2`, ~80MB).
4. Launches the Knowledge Base Studio at `http://localhost:8000`.

---

### Option 2: Manual Setup (Git / All Platforms)

```bash
# 1. Clone repository
git clone https://github.com/AryanSingh64/Mi-RAG.git
cd Mi-RAG

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start local factory microservice
python run_factory.py
```

Open `http://localhost:8000` in your web browser.

---

## Architecture Lifecycle

```text
+-------------------------------------------------------------------------+
|                              INGESTION                                  |
|   PDFs / Word Docs / Images -> PyMuPDF Vector Clustering + RapidOCR    |
|   -> Semantic Chunks with Extracted High-Resolution Diagrams            |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                              EMBEDDING                                  |
|   Local SentenceTransformers (all-MiniLM-L6-v2, 384-dim vectors)       |
|   -> Embedded ChromaDB Vector Store with Metadata Linking               |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                              RETRIEVAL                                  |
|   User Query / Attached Image -> Multi-Turn Attention Reranking         |
|   -> Top-K Relevant Document & Diagram Chunks                           |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                              SYNTHESIS                                  |
|   Local Ollama LLM + Grounding Verification & Citation Checking         |
|   -> Verified Answer + High-Resolution Diagram Evidence Cards           |
+-------------------------------------------------------------------------+
```

---

## Hardware Auto-Tuning Profiles

The engine automatically selects quantization and model parameters based on your available hardware:

| Profile | Compute Target | Recommended LLM | Quantization | Features |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | High VRAM (>= 12GB) | `qwen2.5:7b` / `qwen2.5:14b` | FP16 / Q8_0 | Full visual ensemble & high-res vision |
| **Tier 2** | Mid-Range (6GB - 11GB) | `llama3.2:3b` / `qwen2.5:3b` | Q4_K_M | Fast inference, balanced memory use |
| **Tier 3** | CPU Only / Ultrabook | `qwen2.5:1.5b` / `llama3.2:1b` | Q4_0 | ONNX CPU runtime, multi-threaded batching |

---

## Standalone Turnkey Export

When you index a knowledge base in the studio, click **Download Standalone ZIP** to export a self-contained production package:

```text
standalone_bundle/
|-- vector_db/         # Pre-indexed ChromaDB vector database
|-- images/            # Extracted diagram crops and visual figures
|-- static/assets/     # Logos, icons, and UI assets
|-- server.py          # Standalone FastAPI microservice
|-- index.html         # Hydrated direct chat UI with local storage memory
|-- setup.py           # Dependency verification script
|-- requirements.txt   # Core Python dependencies
|-- run.bat            # Windows 1-click launcher (auto port detection)
|-- run.sh             # Linux/macOS launcher
+-- Dockerfile         # Container deployment configuration
```

### Running the Exported Package

#### Windows
Double-click `run.bat` or run from terminal:
```cmd
run.bat
```

#### Linux / macOS
```bash
chmod +x run.sh
./run.sh
```

#### Docker
```bash
docker compose up --build
```

---

## REST API Reference

The local server exposes standard REST endpoints for external integrations:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/sessions/create` | Creates an isolated session with private vector storage |
| `POST` | `/api/sessions/{id}/upload` | Uploads and parses a PDF, Word document, or image |
| `POST` | `/api/sessions/{id}/chat` | Queries the knowledge base with multi-turn memory |
| `POST` | `/api/sessions/{id}/clear_memory` | Clears conversation memory for the session |
| `GET` | `/api/sessions/{id}/images/{file}` | Serves extracted diagrams and visual crops |
| `GET` | `/api/sessions/{id}/export` | Downloads the turnkey standalone ZIP package |
| `POST` | `/api/shutdown` | Gracefully terminates the local standalone server |

### Example Chat Request (`/api/chat`)

```json
{
  "message": "Explain the architecture described in Figure 3",
  "top_k": 6,
  "history": [
    {"role": "user", "content": "What is SSRL?"},
    {"role": "assistant", "content": "SSRL is Self-Supervised Refinement Learning..."}
  ]
}
```

---

## Requirements

- **Python**: 3.10 or higher
- **Ollama**: Local Ollama runtime ([ollama.com](https://ollama.com))
- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS (Apple Silicon / Intel)

---

## License

Open source under the [MIT License](LICENSE). Built by Aryan Singh.
