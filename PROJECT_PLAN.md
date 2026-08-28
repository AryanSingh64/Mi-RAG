# Autonomous RAG Factory ("RAG-in-a-Box")
## Architectural Blueprint & Implementation Plan

---

## 1. Executive Summary & Vision

The **Autonomous RAG Factory** is a local-first, zero-budget platform that enables users and organizations to:
1. **Upload raw documents** of virtually any format (standard text documents, scanned PDFs, or clicked camera photos/diagrams).
2. **Configure pipeline parameters** (embedding models, local Ollama LLMs, chunking sizes, similarity thresholds, and strict anti-hallucination guardrails).
3. **Automate ingestion & indexing** into a lightweight, portable vector database.
4. **Access an Ephemeral Shareable Portal** protected by a time-limited token (e.g., valid for 2–4 hours) allowing live interactive testing with citations.
5. **Download a Standalone Production Package (ZIP)** containing a turnkey, production-grade RAG application (FastAPI backend + embedded vector store + frontend UI + Docker configuration + local startup scripts).

---

## 2. Constraints & Design Principles

* **Zero-Budget & 100% Free/Open-Source Stack**: No recurring cloud bills, proprietary API dependencies, or paid database instances.
* **Laptop & Edge Friendly**: Runs efficiently on consumer hardware (CPU or moderate GPU) using lightweight embeddings and local Ollama models.
* **Maximum File Compatibility**: Native text extraction with multi-modal vision and local OCR fallbacks for clicked/scanned imagery.
* **Anti-Hallucination by Design**: Deterministic context filtering, source attribution citations, and strict refusal prompts when evidence is missing.
* **Clean Code & Modularity**: Separation of concerns using Domain-Driven Design (DDD) for testability and extensibility.

---

## 3. System Architecture

```
                                  USER INTERFACE
           ┌─────────────────────────────────────────────────────────┐
           │  • Document Ingestion Wizard                            │
           │  • Parameter & Model Selector (Ollama / Local Embeds)   │
           │  • Ephemeral Testing Playground (Live Chat + Citations) │
           │  • One-Click Standalone ZIP Exporter                    │
           └────────────────────────────┬────────────────────────────┘
                                        │
                                        ▼
                             CORE FACTORY BACKEND
  ┌───────────────────────────────────────────────────────────────────────────┐
  │ 1. Ingestion Engine                                                       │
  │    ├─ Native Parsers: PDF, DOCX, TXT, MD, CSV, JSON                      │
  │    └─ Vision & OCR: Ollama Vision (moondream/llama3.2-vision) & RapidOCR  │
  │                                                                           │
  │ 2. Chunking & Indexing                                                    │
  │    ├─ Semantic & Recursive Chunker (Preserves context & hierarchy)        │
  │    ├─ Local Embedder: Sentence-Transformers (BGE-small / MiniLM)          │
  │    └─ Portable Vector Store: ChromaDB / LanceDB (File-backed)             │
  │                                                                           │
  │ 3. Anti-Hallucination & Retrieval Guardrails                              │
  │    ├─ Relevance Threshold Filter (Prunes low-confidence chunks)           │
  │    ├─ Strict Grounded Prompting Engine with Mandatory Citations           │
  │    └─ Zero-Evidence Fallback Detector                                     │
  │                                                                           │
  │ 4. Session & Packaging Manager                                            │
  │    ├─ Ephemeral Session Tokens with TTL Expiration                        │
  │    └─ Standalone "RAG-in-a-Box" Exporter (Zips Backend + DB + Web UI)     │
  └───────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Multi-Modal Ingestion & OCR Strategy

Handling standard documents as well as photographed or scanned notes without cloud dependencies:

| File Category | Extensions | Processing Strategy |
| :--- | :--- | :--- |
| **Standard Documents** | `.pdf`, `.docx`, `.txt`, `.md`, `.csv`, `.json` | Direct text extraction via `pypdf`, `python-docx`, and native text parsers. |
| **Scanned PDFs** | `.pdf` (without text layer) | Automatic fallback: converts pages to images and routes through local OCR/Vision. |
| **Clicked Photos / Images** | `.png`, `.jpg`, `.jpeg`, `.webp` | Dual-mode processing: **Ollama Vision** (e.g., `moondream`, `llama3.2-vision`) for semantic comprehension + **RapidOCR** (CPU-native) for raw text fallback. |

---

## 5. Anti-Hallucination Guardrails Pipeline

To ensure the generated RAG system produces reliable, production-grade responses:

1. **Distance/Similarity Pruning**:
   - Chunks below a configurable cosine similarity score are discarded before reaching the LLM context window.
2. **Strict Grounding Prompt Template**:
   - System instructions explicitly forbid answering from general parametric memory.
   - Requires inline citations for each claim: `[Source: document.pdf, Section: Overview, Chunk #3]`.
3. **Graceful Fallback Handling**:
   - If no chunks pass the confidence threshold, the model deterministically responds:
     > *"The requested information is not available in the provided documentation."*
4. **Attribution & Transparency Output**:
   - Every response returns both the natural language answer and the exact retrieved chunks with confidence scores for user verification.

---

## 6. Ephemeral Shareable Portal & "RAG-in-a-Box" Exporter

### A. Ephemeral Shareable Portal
* Generates a temporary unique link with a configurable Time-To-Live (TTL) (e.g., 2–4 hours).
* Clients can test their specific RAG pipeline live in a web playground without logging in or configuring infrastructure.

### B. "RAG-in-a-Box" Standalone ZIP Bundle
When the user clicks **"Download Deployment Package"**, the factory bundles:
* `data/vector_db/`: Pre-indexed vector store files containing their embedded data.
* `app/`: Self-contained, lightweight FastAPI server.
* `web/`: Pre-configured, responsive chat UI.
* `Dockerfile` & `docker-compose.yml`: Ready for instant containerized deployment.
* `run.bat` (Windows) & `run.sh` (Linux/macOS): One-click local startup scripts.
* `README.md` & `INSTALL_GUIDE.md`: Step-by-step setup documentation tailored to their model selection.

---

## 7. Project Directory Structure

```
a:/RAG/
├── core/                           # Domain & Business Logic
│   ├── ingestion/                  # Document & Image Parsers
│   │   ├── base.py                 # Ingestion Interface
│   │   ├── text_parser.py          # TXT, MD, CSV, JSON
│   │   ├── docx_parser.py          # DOCX parser
│   │   ├── pdf_parser.py           # Native & Scanned PDF parser
│   │   └── vision_ocr.py           # Ollama Vision & Local OCR parser
│   ├── chunking/                   # Text chunking strategies
│   ├── embeddings/                 # SentenceTransformers local wrapper
│   ├── vectorstore/                # ChromaDB file-backed manager
│   ├── guardrails/                 # Anti-hallucination & grounding logic
│   └── llm/                        # Ollama client & prompt generator
│
├── packager/                       # Bundle & ZIP generation
│   ├── templates/                  # Templates for standalone exported app
│   └── exporter.py                 # Bundles code, database, and scripts
│
├── server/                         # Main Factory API & Session Controller
│   ├── api/                        # REST endpoints (upload, build, chat, export)
│   ├── sessions/                   # Ephemeral session store with TTL
│   └── main.py                     # Factory server entry point
│
├── web/                            # Factory Frontend
│   ├── static/                     # CSS, JS, Assets
│   └── templates/                  # Builder UI, Playground UI, Portal UI
│
├── requirements.txt                # Python dependencies
└── README.md                       # Main project documentation
```

---

## 8. Development Roadmap & Milestones

* [ ] **Phase 1: Environment & Core Ingestion Pipeline**
  * Set up dependency environment (`requirements.txt`).
  * Implement multi-format document parsers + image OCR/Vision extractor.
  * Implement recursive chunking and local embedding generation.
* [ ] **Phase 2: Vector Storage & Anti-Hallucination Query Engine**
  * Configure ChromaDB local persistence.
  * Implement Ollama integration with strict grounding prompt templates and citation tracking.
  * Build confidence scoring and fallback detection.
* [ ] **Phase 3: Standalone Packager ("RAG-in-a-Box")**
  * Create export templates (FastAPI standalone runtime + web chat UI + Dockerfile).
  * Build automated ZIP packager that bundles pre-indexed vector data with runtime code.
* [ ] **Phase 4: Web UI & Ephemeral Shareable Portal**
  * Build Ingestion & Configuration Wizard.
  * Build Interactive Testing Playground with live confidence scores and citations.
  * Implement expiring session manager and shareable portal links.
* [ ] **Phase 5: End-to-End Verification & Benchmarking**
  * Test with standard docs, scanned documents, and clicked camera photos.
  * Verify standalone exported ZIP unpack and run workflow.
