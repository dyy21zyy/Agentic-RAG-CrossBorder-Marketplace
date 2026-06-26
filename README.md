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

Current status: Stage 5 real Milvus hybrid retrieval layer.

Implemented through Stage 5:
- fixed scaffold
- core schemas and interfaces
- parser IO utilities
- trademark XML parser
- patent TSV parser
- litigation CSV parser
- policy document parser
- parser scripts 01–04
- logical chunking for trademark, patent, policy, and litigation documents
- chunk build script 05
- DuckDB structured storage
- DuckDB exact lookup APIs
- DuckDB build script 06
- real embedding provider interfaces
- OpenAI-compatible embedding provider
- local sentence-transformer embedding provider
- real Milvus vector store
- local BM25 retriever
- RRF fusion
- lexical/local/API reranker interfaces
- HybridRetriever
- Milvus index build script 07
- local Milvus docker-compose

Still not implemented:
- Agentic RAG workflow
- query classification
- answer generation
- evaluation and ablation

### Stage 5 Milvus and retrieval commands

```bash
docker compose up -d
python scripts/07_build_milvus_index.py --input data/processed/chunks.jsonl --dry-run --report data/processed/milvus_report.json
python scripts/07_build_milvus_index.py --input data/processed/chunks.jsonl --collection-name ip_chunks --overwrite --report data/processed/milvus_report.json
```

FakeEmbeddingProvider is only for tests and smoke runs. Real semantic retrieval requires OpenAI-compatible or local sentence-transformer embeddings. Real Milvus mode requires a running Milvus instance and pymilvus installed. Mock Milvus is only used in unit tests. The Docker Compose stack is for local development and is not required for default tests.

### Stage 2 parser commands

```bash
python scripts/01_parse_trademark_xml.py --input "<TRADEMARK_RAW_DIR>" --output data/processed/trademarks.jsonl --report data/processed/trademark_report.json
python scripts/02_parse_patent_tsv.py --input "<PATENT_RAW_DIR>" --output data/processed/patents.jsonl --report data/processed/patent_report.json
python scripts/03_parse_litigation_csv.py --input "<LITIGATION_RAW_DIR>" --output data/processed/litigation.jsonl --report data/processed/litigation_report.json
python scripts/04_parse_policy_docs.py --input "<POLICY_RAW_DIR>" --output data/processed/policies.jsonl --report data/processed/policy_report.json
```

### Stage 3 chunk build commands

```bash
python scripts/05_build_chunks.py --input data/processed/policies.jsonl --output data/processed/chunks.jsonl --report data/processed/chunk_report.json
python scripts/05_build_chunks.py --input data/processed/trademarks.jsonl --output data/processed/trademark_chunks.jsonl --report data/processed/trademark_chunk_report.json
```

### Stage 4 DuckDB build commands

```bash
python scripts/06_build_duckdb.py --input data/processed/trademarks.jsonl --duckdb-path data/processed/ip.duckdb --report data/processed/duckdb_report.json --overwrite
python scripts/06_build_duckdb.py --input data/processed/all_docs.jsonl --duckdb-path data/processed/ip.duckdb --report data/processed/duckdb_report.json --overwrite
```

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

DuckDB is included in the default installation for Stage 4 structured storage. Optional dependencies for local embeddings, Milvus, PDFs, and HTML parsing remain separated into extras.

## Run tests

```bash
python -m compileall -q src scripts
pytest -q
```

## Data and artifact rule

Do not commit raw data or generated artifacts. Keep full local datasets and generated outputs outside version control. The `.gitignore` excludes `data/raw/`, `data/processed/`, `data/eval/`, DuckDB files, SQLite files, and local environment files.
