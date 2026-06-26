# Agentic RAG for Cross-Border Marketplace Intellectual Property QA

跨境电商知识产权 Agentic RAG 问答系统

## Project goal

This repository will implement an Agentic Retrieval-Augmented Generation (RAG) system for cross-border marketplace intellectual property QA. The final system is intended to help answer questions over trademark, patent, patent-litigation, and marketplace policy evidence while preserving source-aware retrieval and evaluation workflows.

## Data sources

The planned data sources are:

- USPTO Trademark Full Text XML Data
- PatentsView Granted Patent Long Text Data
- Patent Litigation Docket Reports Data
- Temu policy documents

Raw data, generated DuckDB files, vector indexes, and large processed artifacts must not be committed.

## Final architecture overview

The final project structure is designed around these layers:

1. **Ingestion**: parse trademark XML, patent TSV, litigation CSV, and policy documents.
2. **Chunking**: convert normalized source documents into retrieval chunks.
3. **Storage**: persist structured metadata in DuckDB and vectors in Milvus.
4. **Retrieval**: combine BM25, dense retrieval, reciprocal rank fusion, and reranking.
5. **Agent workflow**: classify questions, plan retrieval, route SQL-style needs, evaluate evidence, and synthesize answers.
6. **Evaluation**: run retrieval, answer-quality, and ablation experiments.

## Stage roadmap

- **Stage 0**: complete fixed scaffold only.
- **Stage 1**: core schemas and interfaces.
- **Stage 2**: source parsers and ingestion scripts.
- **Stage 3**: chunking.
- **Stage 4**: DuckDB storage.
- **Stage 5**: Milvus, embeddings, retrieval, and reranking.
- **Stage 6**: Agentic RAG workflow and query CLI.
- **Stage 7**: evaluation and ablation runners.
- **Stage 8**: end-to-end fixtures, integration tests, and final documentation.

## Current status

Stage 1 foundation interfaces.

Implemented:
- fixed project scaffold from Stage 0
- core schemas
- source-type and route validation
- deterministic fake embedding provider for tests and smoke runs
- reranker base interface and NoOpReranker

Not implemented yet:
- source parsers
- logical chunking
- DuckDB storage
- Milvus storage and indexing
- BM25, dense retrieval, RRF
- real embedding providers
- real rerankers
- Agentic RAG workflow
- evaluation and ablation

## Repository structure

```text
Agentic-RAG-CrossBorder-Marketplace/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── docker-compose.yml
├── configs/
├── scripts/
├── src/crossborder_agentic_rag/
└── tests/
```

The full Stage 0 tree is enforced by `tests/test_stage0_scaffold.py`.

## Set up environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Optional dependencies for local embeddings, Milvus, DuckDB, PDFs, and HTML parsing are separated into extras so the default installation remains lightweight.

## Run tests

```bash
python -m compileall -q src scripts
pytest -q
```

## Data and artifact rule

Do not commit raw data or generated artifacts. Keep full local datasets and generated outputs outside version control. The `.gitignore` excludes `data/raw/`, `data/processed/`, `data/eval/`, DuckDB files, SQLite files, and local environment files.
