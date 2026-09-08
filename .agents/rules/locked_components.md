# LOCKED CORE ARCHITECTURE & UI CONTRACT

> [!IMPORTANT]
> **USER INSTRUCTION LOCK (ACTIVE)**:
> The following implementations are **STRICTLY LOCKED** per user directive. Do NOT alter, refactor, replace, or modify any of these components unless the user explicitly requests changes and confirms beforehand.

---

### 1. Contextual Inline Diagram UI & Integration
- **Right-Floated Side Card**:
  - Floats on the right side on desktop (`width: 350px`, `max-width: 46%`, `margin: 0.35rem 0 1rem 1.35rem`).
  - Text and paragraphs flow naturally alongside the image.
  - Minimal border (`1px solid rgba(255, 255, 255, 0.08)`), clean rounded corners (`border-radius: 8px`), and subtle elevated shadow.
  - White inner canvas (`background: #ffffff; padding: 0.35rem;`) so scientific/medical plots have zero ugly black letterboxing.
  - Floating zoom button on hover for full-size modal viewing.
  - Compact bottom caption bar displaying diagram title, provenance page, and match score pill.
- **Smart Section-Anchor Fallback**:
  - Automatically anchors the diagram card directly below the most relevant paragraph when LLMs omit markdown tags.
  - Suppresses duplicate bottom galleries when images are already placed inline.

### 2. Compact Modern Typography
- **Lists**: No stray `<br>` tags inside `<ul>` or `<ol>`. Tight vertical spacing (`li { margin-bottom: 0.25rem; line-height: 1.55; }`).
- **Headings & Floats**: Headings clear floats (`.msg-h1, .msg-h2, .msg-h3 { clear: both; }`) to ensure flawless visual rhythm.
- **Markdown Tables**: Sleek dark-mode pipe tables (`.table-container`, `.markdown-table`) with contrasting headers and soft row borders.

### 3. Hybrid Lexical Search & Recall Guarantee
- **`ChromaVectorStore.keyword_search()`**: Exact substring `$contains` retrieval in ChromaDB to guarantee 100% recall for technical terms and keywords (e.g. "confusion matrix", "Keywords", "OncotypeDX").
- **`AntiHallucinationEngine.filter_relevant_chunks()`**: Lexical match boost ensuring exact query phrase/keyword chunks are NEVER discarded by vector similarity thresholds.

### 4. Exemplar Topic Isolation & Semantic Visual Matching
- **`get_feedback_exemplars()`**: Requires content word `overlap >= 1` to strictly prevent cross-document topic contamination (e.g. AlphaFold leaking into medical papers).
- **`_extract_relevant_images()`**:
  - Suppresses diagrams on purely conceptual queries ("what is a loss function?").
  - Strictly requires subject overlap between visual queries (e.g. "Grad-CAM") and candidate image metadata/text before attaching.
