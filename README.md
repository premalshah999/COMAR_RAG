# COMAR RAG — Hybrid Retrieval-Augmented Generation for the Maryland Code of Regulations

A production-quality RAG system that makes the **Code of Maryland Regulations (COMAR)** accessible through natural-language queries. The system ingests COMAR Titles 15 (Agriculture) and 26 (Environment), indexes **50,827 semantic chunks** using BGE-M3 dense+sparse embeddings in Qdrant, and generates citation-grounded answers via DeepSeek-V3.

> **Authors:** Premal Shah · James Purtilo — University of Maryland, College Park (CMSC 607)

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Running Infrastructure](#running-infrastructure)
  - [Data Ingestion](#data-ingestion)
  - [Starting the API Server](#starting-the-api-server)
  - [Starting the Frontend](#starting-the-frontend)
- [API Reference](#api-reference)
- [Pipeline Deep Dive](#pipeline-deep-dive)
  - [Ingestion Pipeline](#ingestion-pipeline)
  - [Retrieval Pipeline](#retrieval-pipeline)
  - [Generation Pipeline](#generation-pipeline)
- [Testing](#testing)
- [Configuration Reference](#configuration-reference)
- [License](#license)

---

## Features

- **Hybrid Retrieval** — Dense semantic + sparse BM25-like search fused via Reciprocal Rank Fusion (RRF)
- **Knowledge Graph Expansion** — NetworkX graph adds cross-referenced regulations and chapter definitions to retrieval context
- **Cross-Encoder Reranking** — BAAI/bge-reranker-v2-m3 rescores candidates for precision
- **Seven-Class Intent Classifier** — Zero-latency keyword/regex router directs queries (conversational, citation_lookup, definition, compliance, overview, enforcement, general)
- **Citation Verification** — Post-generation check flags any COMAR citation not grounded in retrieved context
- **Multi-Turn Conversations** — Bounded LRU conversation store with vague-query expansion for follow-ups
- **Streaming Responses** — Server-Sent Events (SSE) for token-by-token delivery
- **LangGraph Agentic Pipeline** — State-machine with direct citation lookup, query rewriting loop, and verification
- **React Frontend** — Dark/light theme, sidebar history, citation cards, health dashboard
- **Stub Mode** — Full system exercisable without an API key (deterministic fallback responses)

---

## Architecture Overview

```
┌─────────────┐    SSE/JSON     ┌──────────────┐     Qdrant       ┌────────────────┐
│   React UI  │ ◄─────────────► │  FastAPI API  │ ◄──────────────► │  Qdrant Vector │
│  (Vite/TS)  │                 │  (Streaming)  │                  │     Database   │
└─────────────┘                 └──────┬───────┘                  └────────────────┘
                                       │
                              ┌────────┴────────┐
                              │                 │
                    ┌─────────▼──────┐  ┌───────▼─────────┐
                    │ Intent Router  │  │ Graph Expander   │
                    │ (7 classes)    │  │ (NetworkX Graph) │
                    └─────────┬──────┘  └───────┬─────────┘
                              │                 │
                    ┌─────────▼─────────────────▼──────────┐
                    │        LLM Generation Layer          │
                    │  DeepSeek-V3 / Anthropic / OpenAI    │
                    └─────────┬────────────────────────────┘
                              │
                    ┌─────────▼──────┐
                    │   Citation     │
                    │   Verifier     │
                    └────────────────┘
```

---

## Project Structure

```
COMAR RAG/
│
├── api/                          # FastAPI backend
│   ├── main.py                   # App factory, CORS, router mounting, SPA serving
│   ├── config.py                 # Pydantic Settings from .env (LLM, Qdrant, Neo4j, embeddings)
│   ├── models.py                 # Request/response Pydantic models (ChatRequest, Source, HealthResponse, etc.)
│   ├── routes/
│   │   ├── chat.py               # POST /api/chat — SSE streaming endpoint with conversation memory
│   │   └── health.py             # GET /api/health, GET /api/stats — system status & corpus statistics
│   └── services/
│       ├── intent.py             # Zero-latency rule-based 7-class intent classifier
│       ├── llm.py                # Async LLM streaming (DeepSeek → Anthropic → OpenAI → stub fallback)
│       └── retriever.py          # Production retrieval: embed → hybrid search → graph expand → rerank → format
│
├── ingestion/                    # Data ingestion pipeline
│   ├── run_ingestion.py          # Master CLI orchestrator (7-step pipeline)
│   ├── fetch_comar.py            # Download COMAR XML from GitHub (maryland-dsd/law-xml)
│   ├── xml_parser.py             # Parse XML hierarchy → structured regulation dicts (xi:include resolution)
│   ├── chunker.py                # Semantic chunking: primary + subsection + definition extraction
│   ├── graph_builder.py          # Build NetworkX DiGraph (Title → Subtitle → Chapter → Regulation + REFERENCES)
│   ├── embedder.py               # BGE-M3 wrapper (dense 1024d + sparse BM25 + ColBERT vectors)
│   └── qdrant_uploader.py        # Create Qdrant collection with named vectors & upload chunks
│
├── retrieval/                    # Three-stage retrieval pipeline
│   ├── __init__.py               # COMARRetriever — orchestrates hybrid → graph expand → rerank
│   ├── hybrid_retriever.py       # Dense + sparse Qdrant search fused with RRF (k=60)
│   ├── graph_expander.py         # Walk knowledge graph for cross-refs and chapter definitions
│   └── reranker.py               # Cross-encoder reranking with BAAI/bge-reranker-v2-m3
│
├── pipeline/                     # LangGraph agentic RAG pipeline
│   ├── langgraph_pipeline.py     # StateGraph: route → direct_lookup/hybrid → build_context → generate → verify
│   ├── router.py                 # QueryRouter: LLM + heuristic query classification (5 categories)
│   ├── prompts.py                # System prompt, router prompt, query rewrite prompt
│   └── citation_verifier.py      # Post-generation citation hallucination detection
│
├── frontend/                     # React + TypeScript + Tailwind UI
│   ├── index.html                # HTML entry point
│   ├── package.json              # Dependencies: React 18, Framer Motion, Lucide icons, Tailwind
│   ├── vite.config.ts            # Vite config with /api proxy to backend
│   ├── tailwind.config.js        # Tailwind theme configuration
│   ├── tsconfig.json             # TypeScript config
│   └── src/
│       ├── main.tsx              # React DOM entry
│       ├── App.tsx               # Root component with ErrorBoundary, page routing, health polling
│       ├── index.css             # Global styles & Tailwind imports
│       ├── types.ts              # TypeScript interfaces (Message, Source, Conversation, HealthStatus, etc.)
│       ├── hooks/
│       │   ├── useChat.ts        # Chat state management, SSE streaming, conversation switching
│       │   └── useTheme.ts       # Dark/light theme toggle with localStorage persistence
│       └── components/
│           ├── Landing.tsx       # Landing page with animated entry
│           ├── ChatWindow.tsx    # Scrollable message container
│           ├── MessageBubble.tsx # User/assistant message rendering with markdown
│           ├── CitationCard.tsx  # Expandable citation source cards with metadata
│           ├── InputBar.tsx      # Text input with send/stop controls
│           ├── Sidebar.tsx       # Conversation list, title filter, new chat button
│           ├── TopBar.tsx        # Header bar with health status indicator
│           ├── Header.tsx        # Reusable header component
│           └── ThemeToggle.tsx   # Dark/light mode switcher
│
├── tests/                        # Pytest test suite
│   ├── conftest.py               # Shared pytest configuration
│   ├── test_pipeline.py          # End-to-end LangGraph pipeline tests (MockLLM + MockRetriever)
│   └── test_retrieval.py         # Integration tests for hybrid retrieval, graph expansion, reranking
│
├── colab/
│   └── comar_ingestion.ipynb     # Google Colab notebook for cloud-based ingestion
│
├── data/
│   ├── definitions.json          # Extracted COMAR term → definition lookup (generated by ingestion)
│   └── comar_graph.pkl           # Pickled NetworkX knowledge graph (generated by ingestion)
│
├── docker-compose.yml            # Qdrant + Neo4j infrastructure containers
├── requirements.txt              # Python dependencies
├── .env.example                  # Template for environment variables
├── .gitignore                    # Git ignore rules
└── REPORT.md                     # Full academic report (research questions, methodology, evaluation)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | DeepSeek-V3 (primary), Anthropic Claude, OpenAI GPT — via OpenAI-compatible API |
| **Embeddings** | BAAI/BGE-M3 (dense 1024d + sparse BM25 + ColBERT) |
| **Reranker** | BAAI/bge-reranker-v2-m3 (cross-encoder) |
| **Vector DB** | Qdrant (named dense + sparse vectors, payload indexes) |
| **Knowledge Graph** | NetworkX DiGraph (CONTAINS, REFERENCES, DEFINES edges) |
| **Pipeline** | LangGraph (StateGraph with conditional edges, rewrite loop) |
| **Backend** | FastAPI + Uvicorn (async SSE streaming) |
| **Frontend** | React 18 + TypeScript + Vite + Tailwind CSS + Framer Motion |
| **Infrastructure** | Docker Compose (Qdrant, Neo4j) |
| **Testing** | Pytest + pytest-asyncio |

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** and npm
- **Docker** and Docker Compose
- A **DeepSeek API key** (or Anthropic/OpenAI key — the system falls back gracefully; also runs in stub mode without any key)

### Installation

```bash
# Clone the repository
git clone https://github.com/premalshah999/COMAR_RAG.git
cd COMAR_RAG

# Create a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### Environment Variables

Copy the template and fill in your API keys:

```bash
cp .env.example .env
```

Key variables:

| Variable | Description | Default |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API key (primary LLM) | — |
| `LLM_PROVIDER` | `deepseek`, `anthropic`, or `openai` | `deepseek` |
| `LLM_MODEL` | Model name | `deepseek-chat` |
| `QDRANT_HOST` | Qdrant hostname | `localhost` |
| `QDRANT_PORT` | Qdrant port | `6333` |
| `QDRANT_COLLECTION` | Collection name | `comar_regulations` |
| `NEO4J_URI` | Neo4j Bolt URI | `bolt://localhost:7687` |
| `NEO4J_PASSWORD` | Neo4j password | `comar_password` |
| `BGE_M3_DEVICE` | PyTorch device (`cpu`, `cuda`, `mps`) | `cpu` |
| `DATA_DIR` | XML cache directory | `./data/xml_cache` |

### Running Infrastructure

Start Qdrant and Neo4j via Docker Compose:

```bash
docker compose up -d
```

This starts:
- **Qdrant** on ports 6333 (HTTP) and 6334 (gRPC)
- **Neo4j** on ports 7474 (browser) and 7687 (Bolt)

### Data Ingestion

Run the 7-step ingestion pipeline to populate the vector database:

```bash
python ingestion/run_ingestion.py
```

Pipeline steps:
1. **Fetch** — Download COMAR XML from GitHub (`maryland-dsd/law-xml`)
2. **Parse** — Resolve xi:includes, walk Title → Subtitle → Chapter → Section hierarchy
3. **Chunk** — Create primary, subsection, and definition chunks with breadcrumbs
4. **Graph** — Build NetworkX knowledge graph (nodes: titles/subtitles/chapters/regulations; edges: CONTAINS/REFERENCES/DEFINES)
5. **Embed** — Initialize BGE-M3 model
6. **Upload** — Upsert all chunks with dense + sparse vectors to Qdrant
7. **Save** — Write `definitions.json` and `comar_graph.pkl` to `data/`

Options:
```bash
python ingestion/run_ingestion.py --refresh        # Force re-download XML
python ingestion/run_ingestion.py --skip-fetch     # Use cached XML
python ingestion/run_ingestion.py --titles 15      # Process only Title 15
python ingestion/run_ingestion.py --skip-upload     # Build chunks/graph only
```

### Starting the API Server

```bash
uvicorn api.main:app --reload --port 8000
```

- API docs: http://localhost:8000/api/docs
- Health check: http://localhost:8000/api/health

### Starting the Frontend

```bash
cd frontend
npm run dev
```

Opens at http://localhost:5173 with hot-reload. The Vite dev server proxies `/api` requests to the backend on port 8000.

For production:
```bash
cd frontend && npm run build
# The built SPA in frontend/dist/ is automatically served by the FastAPI app
```

---

## API Reference

### `POST /api/chat`

Stream a response to a user message via Server-Sent Events.

**Request body:**
```json
{
  "message": "What permits are needed for pesticide storage?",
  "conversation_id": "optional-uuid",
  "filters": { "title_num": ["15"] },
  "top_k": 10
}
```

**SSE events:**
```
data: {"token": "COMAR", "done": false}
data: {"token": " 15.05", "done": false}
...
data: {"token": "", "done": true, "sources": [...], "conversation_id": "...", "mode": "live"}
```

### `GET /api/health`

Returns system status:
```json
{
  "status": "ok",
  "qdrant_connected": true,
  "qdrant_points": 50827,
  "qdrant_collection": "comar_regulations",
  "llm_ready": true,
  "llm_model": "deepseek-chat"
}
```

### `GET /api/stats`

Returns corpus statistics (regulation count, graph nodes/edges, definition count, etc.).

---

## Pipeline Deep Dive

### Ingestion Pipeline

```
GitHub (maryland-dsd/law-xml)
        │
        ▼
  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
  │  fetch_comar │ ──► │  xml_parser  │ ──► │     chunker      │
  │  (Git API)   │     │ (xi:include  │     │ (primary +       │
  │              │     │   + lxml)    │     │  subsection +    │
  └──────────────┘     └──────────────┘     │  definitions)    │
                                            └────────┬─────────┘
                                                     │
                                     ┌───────────────┼───────────────┐
                                     ▼               ▼               ▼
                              ┌────────────┐  ┌──────────────┐  ┌────────────────┐
                              │   graph    │  │   embedder   │  │ definitions    │
                              │  builder   │  │  (BGE-M3)    │  │    .json       │
                              │ (NetworkX) │  └──────┬───────┘  └────────────────┘
                              └────────────┘         │
                                                     ▼
                                              ┌──────────────┐
                                              │   qdrant     │
                                              │  uploader    │
                                              └──────────────┘
```

**Key modules:**

| Module | Responsibility |
|---|---|
| `fetch_comar.py` | Downloads XML files from GitHub via Git Trees API; caches locally; handles rate limits |
| `xml_parser.py` | Resolves `xi:include` references recursively; extracts regulation metadata, text, cross-refs, effective dates |
| `chunker.py` | Creates 3 chunk types: primary (full regulation + breadcrumb), subsection (paragraph splits >600 tokens), definitions (term/definition extraction) |
| `graph_builder.py` | Builds a 4-level hierarchy graph with CONTAINS, REFERENCES, and DEFINES edges |
| `embedder.py` | Wraps BAAI/bge-m3 for dense (1024d), sparse (BM25), and ColBERT vectors in a single forward pass |
| `qdrant_uploader.py` | Creates collection with named vectors + payload indexes; upserts in batches with deterministic UUIDs |

### Retrieval Pipeline

The three-stage retrieval pipeline is orchestrated by `COMARRetriever`:

```
Query ──► HybridRetriever (dense + sparse → RRF fusion, top-100)
              │
              ▼
          GraphExpander (add cross-refs + chapter definitions)
              │
              ▼
          Reranker (cross-encoder → top-8)
```

1. **Hybrid Retrieval** (`hybrid_retriever.py`) — Embeds the query with BGE-M3, runs parallel dense ANN + sparse BM25 searches on Qdrant, fuses with RRF (k=60)
2. **Graph Expansion** (`graph_expander.py`) — Follows REFERENCES edges and pulls .01 Definitions chunks for each chapter represented
3. **Reranking** (`reranker.py`) — BAAI/bge-reranker-v2-m3 cross-encoder scores each (query, chunk) pair; returns the top-N

### Generation Pipeline

The API's generation flow (`api/services/`):

1. **Intent Classification** (`intent.py`) — Rule-based classifier determines one of 7 intents: `conversational`, `citation_lookup`, `definition`, `compliance`, `overview`, `enforcement`, `general`
2. **Retrieval** (`retriever.py`) — Runs the full retrieval pipeline (or direct citation lookup for COMAR XX.XX.XX patterns); applies title filters; deduplicates
3. **LLM Streaming** (`llm.py`) — Selects intent-specific system prompt, builds context from retrieved sources, streams tokens via async generator
4. **Citation Verification** (`citation_verifier.py`) — Extracts COMAR citations from the response, cross-checks against retrieved context, flags unverified citations

**LangGraph Pipeline** (`pipeline/langgraph_pipeline.py`) — Alternative agentic pipeline with a state machine:

```
START → route_query → direct_lookup (if citation) or hybrid_retrieve
                                       │
                                  build_context
                                       │
                                    generate ──► (rewrite loop, max 2 iterations)
                                       │
                                    verify → END
```

---

## Testing

```bash
# Run all tests (requires Qdrant running with data)
pytest tests/ -v

# Run pipeline tests only (uses MockLLM + MockRetriever — no ML models needed)
pytest tests/test_pipeline.py -v

# Run retrieval integration tests (requires BGE-M3 + populated Qdrant)
pytest tests/test_retrieval.py -v
```

**Test architecture:**
- `test_pipeline.py` — Uses `MockLLM` (deterministic response builder) and `MockRetriever` (real Qdrant data, no embedding models) for fast end-to-end tests (~1-3 min)
- `test_retrieval.py` — Full integration tests for hybrid retrieval, citation lookup, graph expansion, and reranker ordering

---

## Configuration Reference

All configuration is managed through environment variables (loaded via `.env`). See [`.env.example`](.env.example) for the full template.

The `api/config.py` module uses Pydantic Settings to validate and expose configuration with type-safe defaults. Key computed properties:
- `Settings.qdrant_ready` — True when Qdrant is reachable and the collection has points
- `Settings.llm_ready` — True when at least one LLM API key is configured

---

## License

This project was developed as a research prototype for CMSC 607 at the University of Maryland, College Park. The COMAR source data is published by the Maryland Division of State Documents under CC0/CC-BY-SA via the [maryland-dsd/law-xml](https://github.com/maryland-dsd/law-xml) repository.
