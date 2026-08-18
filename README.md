<p align="center">
  <h1 align="center">Smart Document Insights</h1>
  <p align="center">
    <em>Advanced RAG pipeline untuk analisis dokumen kompleks.</em>
  </p>
</p>

<p align="center">
  <a href="https://github.com/Maventlabs/smart-document-insights/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  </a>
  <a href="https://github.com/Maventlabs/smart-document-insights/stargazers">
    <img src="https://img.shields.io/github/stars/Maventlabs/smart-document-insights.svg?style=social" alt="Stars">
  </a>
  <a href="https://github.com/Maventlabs/smart-document-insights/network/members">
    <img src="https://img.shields.io/github/forks/Maventlabs/smart-document-insights.svg?style=social" alt="Forks">
  </a>
  <a href="https://github.com/Maventlabs/smart-document-insights/issues">
    <img src="https://img.shields.io/github/issues/Maventlabs/smart-document-insights.svg" alt="Issues">
  </a>
  <a href="https://github.com/Maventlabs/smart-document-insights/pulls">
    <img src="https://img.shields.io/github/issues-pr/Maventlabs/smart-document-insights.svg" alt="Pull Requests">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/LangChain-00C58E.svg?style=flat&logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/NVIDIA_NIM-76B900.svg?style=flat&logo=nvidia&logoColor=white" alt="NVIDIA NIM">
  <img src="https://img.shields.io/badge/Chroma-FFBABA.svg?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDJMMyA3djEwbDkgNSA5LTVIN0wxMiAyeiIgZmlsbD0iIzAwMCIvPjwvc3ZnPg==&logoColor=white" alt="Chroma">
  <img src="https://img.shields.io/badge/RAG-Hybrid-FF6B6B.svg?style=flat" alt="RAG">
</p>

<br>

---

<br>

## Arsitektur Sistem

```mermaid
graph TB
    subgraph Input
        A[PDF / TXT / DOCX]
    end

    subgraph "Document Pipeline"
        B[Parser]
        C[Structure Detector]
        D[Metadata Extractor]
        E[Semantic Chunker]
        F[Embeddings<br/>nemotron-3-embed-1b]
        G[Chroma Vector Store]
    end

    subgraph "Query Pipeline"
        H[User Query]
        I[Query Rewriter]
        J{Hybrid Retrieval}
        K[Semantic Search<br/>Vector / Chroma]
        L[Keyword Search<br/>BM25]
        M[Reciprocal Rank<br/>Fusion]
        N[Cross-Encoder<br/>Reranker]
        O[Context Selection]
        P[LLM<br/>NVIDIA NIM]
    end

    A --> B --> C --> D --> E --> F --> G
    H --> I --> J
    J --> K
    J --> L
    K --> M
    L --> M
    M --> N --> O --> P

    style A fill:#1a1a2e,color:#fff,stroke:#e94560
    style P fill:#76b900,color:#000,stroke:#fff
    style J fill:#16213e,color:#fff,stroke:#e94560
    style G fill:#0f3460,color:#fff,stroke:#533483
```

<br>

## RAG Pipeline Detail

```mermaid
flowchart LR
    subgraph "Stage 1: Ingest"
        A1[Load] --> A2[Parse]
        A2 --> A3[Structure]
        A3 --> A4[Metadata]
        A4 --> A5[Chunk]
        A5 --> A6[Embed]
    end

    subgraph "Stage 2: Retrieve"
        B1[Rewrite] --> B2{Hybrid}
        B2 --> B3[Vector]
        B2 --> B4[BM25]
        B3 --> B5[Fusion]
        B4 --> B5
        B5 --> B6[Rerank]
    end

    subgraph "Stage 3: Generate"
        C1[Context] --> C2[Prompt]
        C2 --> C3[LLM]
        C3 --> C4[Answer]
    end

    A6 -.-> B1
    B6 -.-> C1

    style A6 fill:#533483,color:#fff
    style B6 fill:#e94560,color:#fff
    style C3 fill:#76b900,color:#000
```

<br>

## Tech Stack

```mermaid
graph LR
    A[Streamlit] --> B[LangChain Classic]
    B --> C[NVIDIA NIM API]
    B --> D[ChromaDB]
    B --> E[rank_bm25]
    B --> F[sentence-transformers]
    C --> G[Nemotron 3 Ultra 550B]
    C --> H[nemotron-3-embed-1b]

    style A fill:#FF4B4B,color:#fff
    style C fill:#76b900,color:#000
    style G fill:#0f3460,color:#fff
    style H fill:#0f3460,color:#fff
```

<br>

## Fitur

| Fitur | Deskripsi |
|-------|-----------|
| **Hybrid Retrieval** | Semantic (vector) + Keyword (BM25) dengan Reciprocal Rank Fusion |
| **Cross-Encoder Reranking** | Precisi ranking pakai `ms-marco-MiniLM` |
| **Query Rewriting** | Pronoun resolution + query expansion otomatis |
| **Document Parsing** | Structure-aware: detect headings, tables, sections |
| **Semantic Chunking** | Respect sentence/paragraph boundaries |
| **Multi-Format** | PDF, TXT, DOCX |
| **7 Model NIM** | Nemotron 3 Ultra, Lightning 3.5, Muse Glimmer, dll |
| **Export** | Chat, ringkasan, dan insights ke Markdown |

<br>

## Model yang Tersedia

| Model | Type | ID |
|-------|------|-----|
| Nemotron 3 Ultra 550B | Terkuat | `nvidia/nemotron-3-ultra-550b-a55b` |
| Nemotron 3.5 Lightning 30B | Seimbang | `nvidia/nemotron-3.5-lightning-30b-a3b` |
| Meta Muse Glimmer 30B | Kreatif | `meta/muse-glimmer-30b` |
| StepFun Step 3.7 Flash | Cepat | `stepfun-ai/step-3.7-flash` |
| Poolside Laguna XS 2.1 | Ringan | `poolside/laguna-xs-2.1` |
| Thinking Machines Inkling | — | `thinkingmachines/inkling` |
| **Embedding: Nemotron 3 Embed 1B** | RAG | `nvidia/nemotron-3-embed-1b` |

<br>

## Struktur Proyek

```
smart-document-insights/
├── app.py                         # Entry point
├── src/smart_doc/
│   ├── config.py                  # Konfigurasi
│   ├── core/
│   │   ├── document.py            # Document loading
│   │   ├── document_parser.py     # Enhanced parsing + metadata
│   │   ├── chunker.py             # Semantic chunking
│   │   ├── embeddings.py          # NVIDIA NIM embeddings
│   │   ├── rag.py                 # RAG chain + LLM
│   │   ├── retriever.py           # Hybrid: vector + BM25
│   │   ├── reranker.py            # Cross-encoder reranking
│   │   ├── query_rewriter.py      # Query expansion
│   │   ├── pipeline.py            # Full pipeline orchestrator
│   │   └── prompts.py             # System prompts
│   ├── ui/
│   │   ├── sidebar.py             # Sidebar config
│   │   ├── chat.py                # Chat tab
│   │   ├── summary.py             # Summary tab
│   │   ├── insights.py            # Key insights tab
│   │   └── stats.py               # Document stats
│   └── utils/
│       ├── file.py                # File helpers
│       └── export.py              # Export to Markdown
├── tests/                         # 20 unit tests
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```

<br>

## Instalasi

```bash
git clone https://github.com/Maventlabs/smart-document-insights.git
cd smart-document-insights
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

<br>

## Menjalankan

```bash
streamlit run app.py
```

Buka `http://localhost:8501` lalu:

1. Masukkan **NVIDIA NIM API Key** (gratis di [build.nvidia.com](https://build.nvidia.com))
2. Upload dokumen (PDF/TXT/DOCX)
3. Pilih model → mulai tanya

<br>

## Deploy

### Streamlit Community Cloud (gratis)

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Login GitHub → New app → Pilih repo → Deploy

### Docker

```bash
docker build -t smart-doc-insights .
docker run -p 8501:8501 smart-doc-insights
```

<br>

## Testing

```bash
python -m unittest discover tests -v
```

<br>

## Catatan Keamanan

- API key tidak disimpan permanen
- Dokumen diproses di memori (in-memory)
- Semua model NIM gratis tanpa kartu kredit
- Rate limit: 40 requests/menit

<br>

---

<p align="center">
  <sub>Dibangun dengan NVIDIA NIM + LangChain + Streamlit</sub>
</p>
