# Agentic RAG for Cross-Border Marketplace Intellectual Property QA

跨境电商知识产权 Agentic RAG 问答系统

## Project overview

This project is an Agentic RAG system for cross-border marketplace intellectual property QA over trademark, patent, patent litigation, and marketplace policy evidence. It is designed for source-aware answers to questions such as:

- trademark class / goods lookup
- patent claim explanation
- Temu policy question answering
- patent litigation lookup
- multi-source IP risk analysis

## What this MVP does

- Parses raw trademark XML, patent TSV, litigation CSV, and policy docs.
- Normalizes records into JSONL documents.
- Builds logical `EvidenceChunk` records.
- Builds a DuckDB structured lookup database.
- Builds a Milvus vector index in real mode or dry-run mode.
- Supports local BM25 retrieval.
- Supports dense retrieval through Milvus.
- Supports RRF fusion and reranking.
- Runs a deterministic Agentic RAG workflow.
- Produces adaptive answers with citations.
- Runs evaluation and ablation experiments.
- Includes fixture-based end-to-end tests.

## What this MVP does not do

- It is not legal advice.
- It does not guarantee infringement determination.
- It does not include a production web UI.
- It does not expose an API server.
- It does not train or fine-tune models.
- It does not include full OCR/image trademark analysis.
- It does not use an external LLM judge for faithfulness.
- It does not automatically download USPTO or PatentsView full datasets.

## Data sources

The intended evidence sources are:

- USPTO Trademark Full Text XML Data
- PatentsView Granted Patent Long Text Data
- Patent Litigation Docket Reports Data
- Temu policy documents

Example local source paths, for documentation only; these are not hard-coded in Python:

```text
C:\Users\dyy21\OneDrive\TJ\工作\Code\Rag\Rag_document\Trademark Full Text XML Data
C:\Users\dyy21\OneDrive\TJ\工作\Code\Rag\Rag_document\PatentsView Granted Patent Long Text Data
C:\Users\dyy21\OneDrive\TJ\工作\Code\Rag\Rag_document\Patent Litigation Docket Reports Data
C:\Users\dyy21\OneDrive\TJ\工作\Code\Rag\Rag_document\Temu
```

Raw data, generated DuckDB files, vector indexes, and large processed artifacts must not be committed.

## System architecture

1. **Ingestion** parses trademark XML, patent TSV, litigation CSV, and policy documents.
2. **Normalization** writes source-typed JSONL documents.
3. **Chunking** converts normalized documents into logical evidence chunks.
4. **Structured storage** loads fields into DuckDB for exact lookup.
5. **Retrieval** combines local BM25, optional Milvus dense retrieval, RRF fusion, and reranking.
6. **Agent workflow** classifies the query, plans retrieval, routes SQL-style lookups, evaluates evidence, and synthesizes adaptive answers.
7. **Evaluation** computes deterministic retrieval, routing, answer proxy, and ablation metrics.

## Repository structure

```text
Agentic-RAG-CrossBorder-Marketplace/
├── README.md
├── pyproject.toml
├── .env.example
├── docker-compose.yml
├── configs/
├── scripts/                    # scripts 01-10 for pipeline, query, eval, ablation
├── src/crossborder_agentic_rag/ # ingestion, storage, retrieval, agents, evaluation
└── tests/                      # unit tests and fixture E2E tests
```

## Installation

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
copy .env.example .env
```

Optional dependencies:

```bash
python -m pip install -e '.[milvus]'
python -m pip install -e '.[local]'
python -m pip install -e '.[reranker]'
```

## Environment variables

Key environment variables include:

- `EMBEDDING_PROVIDER`
- `EMBEDDING_API_KEY`
- `EMBEDDING_API_BASE`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIM`
- `RERANKER_PROVIDER`
- `MILVUS_URI`
- `MILVUS_COLLECTION_NAME`
- `DUCKDB_PATH`
- `TRADEMARK_RAW_DIR`
- `PATENT_RAW_DIR`
- `LITIGATION_RAW_DIR`
- `POLICY_RAW_DIR`
- `MAX_RETRIEVAL_ITERATIONS`

FakeEmbeddingProvider is only for tests and smoke runs. Real semantic retrieval requires OpenAI-compatible or local sentence-transformer embeddings.

## Local fixture quickstart

Run the complete Stage 8 fixture pipeline test:

```bash
pytest -q tests/test_stage8_end_to_end.py
```

Manual fixture path examples use `tests/fixtures/e2e` and a temporary or ignored output directory such as `data/processed`.

## Full-data local path examples

Use the environment variables above or substitute quoted paths in commands. Example Windows source folders are documented under **Data sources**. Do not commit full raw datasets or generated artifacts.

## Step-by-step pipeline commands

### Parse trademark

```bash
python scripts/01_parse_trademark_xml.py --input "<TRADEMARK_RAW_DIR>" --output data/processed/trademarks.jsonl --report data/processed/trademark_report.json
```

### Parse patent

```bash
python scripts/02_parse_patent_tsv.py --input "<PATENT_RAW_DIR>" --output data/processed/patents.jsonl --report data/processed/patent_report.json
```

### Parse litigation

```bash
python scripts/03_parse_litigation_csv.py --input "<LITIGATION_RAW_DIR>" --output data/processed/litigation.jsonl --report data/processed/litigation_report.json
```

### Parse policy

```bash
python scripts/04_parse_policy_docs.py --input "<POLICY_RAW_DIR>" --output data/processed/policies.jsonl --report data/processed/policy_report.json
```

### Combine JSONL files

Concatenate the four normalized document files into:

```text
data/processed/all_docs.jsonl
```

Example:

```bash
cat data/processed/trademarks.jsonl data/processed/patents.jsonl data/processed/litigation.jsonl data/processed/policies.jsonl > data/processed/all_docs.jsonl
```

### Build chunks

```bash
python scripts/05_build_chunks.py --input data/processed/all_docs.jsonl --output data/processed/chunks.jsonl --report data/processed/chunk_report.json
```

### Build DuckDB

```bash
python scripts/06_build_duckdb.py --input data/processed/all_docs.jsonl --duckdb-path data/processed/ip.duckdb --report data/processed/duckdb_report.json --overwrite
```

### Build Milvus dry-run

```bash
python scripts/07_build_milvus_index.py --input data/processed/chunks.jsonl --dry-run --report data/processed/milvus_report.json
```

### Build Milvus real mode

```bash
docker compose up -d
python scripts/07_build_milvus_index.py --input data/processed/chunks.jsonl --collection-name ip_chunks --embedding-provider local --overwrite --report data/processed/milvus_report.json
```

### Run query CLI

```bash
python scripts/08_run_query_cli.py "What does Temu policy say about trademark infringement?" --duckdb-path data/processed/ip.duckdb --chunks-path data/processed/chunks.jsonl --output-json
```

### Run evaluation

```bash
python scripts/09_run_eval.py --eval-file tests/fixtures/e2e/eval/eval_queries.jsonl --output-dir data/eval --duckdb-path data/processed/ip.duckdb --chunks-path data/processed/chunks.jsonl
```

### Run ablation

```bash
python scripts/10_run_ablation.py --eval-file tests/fixtures/e2e/eval/eval_queries.jsonl --output-dir data/eval/ablation --duckdb-path data/processed/ip.duckdb --chunks-path data/processed/chunks.jsonl --experiments bm25_only,hybrid_rrf,no_reranker
```

## Milvus local development

Real Milvus mode requires a running Milvus instance, pymilvus installed, and real embeddings configured with `--embedding-provider openai-compatible` or `--embedding-provider local`. Mock Milvus is only used in unit tests. Dry-run mode does not insert into Milvus. Dry-run mode should not be interpreted as successful vector indexing.

For local development:

```bash
docker compose up -d
python scripts/07_build_milvus_index.py --input data/processed/chunks.jsonl --collection-name ip_chunks --embedding-provider local --overwrite --report data/processed/milvus_report.json
```

## Query CLI examples

```bash
python scripts/08_run_query_cli.py "Which Nice classes does MERCEDES belong to?" --duckdb-path data/processed/ip.duckdb --chunks-path data/processed/chunks.jsonl --output-json

python scripts/08_run_query_cli.py "What does Temu policy say about trademark infringement?" --duckdb-path data/processed/ip.duckdb --chunks-path data/processed/chunks.jsonl --output-json

python scripts/08_run_query_cli.py "Can I sell a phone case using the MERCEDES logo on Temu?" --duckdb-path data/processed/ip.duckdb --chunks-path data/processed/chunks.jsonl --output-json

python scripts/08_run_query_cli.py "Summarize litigation history for patent US1234567." --duckdb-path data/processed/ip.duckdb --chunks-path data/processed/chunks.jsonl --output-json
```

Only risk_analysis answers include Risk Level. Plain policy questions about infringement are policy answers, not risk analysis.

## Evaluation

```bash
python scripts/09_run_eval.py --eval-file tests/fixtures/e2e/eval/eval_queries.jsonl --output-dir data/eval --duckdb-path data/processed/ip.duckdb --chunks-path data/processed/chunks.jsonl
```

Evaluation metrics are deterministic and do not call external LLM judges. FaithfulnessProxy is a heuristic, not a human-level factuality evaluator. Demo mode uses fixtures and fake embeddings. Full evaluation requires processed data and retrieval backends.

## Ablation

```bash
python scripts/10_run_ablation.py --eval-file tests/fixtures/e2e/eval/eval_queries.jsonl --output-dir data/eval/ablation --duckdb-path data/processed/ip.duckdb --chunks-path data/processed/chunks.jsonl --experiments bm25_only,hybrid_rrf,no_reranker
```

Ablation experiments must actually change retrieval/reranking/source configuration.

## Real-data readiness checklist

Before running on full datasets:

- Set `TRADEMARK_RAW_DIR`, `PATENT_RAW_DIR`, `LITIGATION_RAW_DIR`, and `POLICY_RAW_DIR` to local raw-data locations.
- Run parser scripts 01-04 for trademark, patent, litigation, and policy inputs.
- Combine normalized JSONL files into `data/processed/all_docs.jsonl`.
- Run chunking script 05.
- Run DuckDB build script 06.
- Run Milvus dry-run script 07 first to validate chunk loading and embedding dimensions without insertion.
- Start Milvus with docker compose for real vector indexing.
- Use real embeddings for semantic retrieval; fake embeddings are for tests and smoke runs only.
- Run the query CLI.
- Run evaluation and ablation after processed data and retrieval backends are available.

Full-data execution may take time and disk space. Do not commit raw or processed full-data artifacts.

## Testing

```bash
python -m compileall -q src scripts
pytest -q
pytest -q tests/test_stage8_end_to_end.py
rg -n "^(<<<<<<<|=======|>>>>>>>)" -S . || true
```

## Known limitations

- Trademark XML field coverage may need expansion for all USPTO variants.
- Patent TSV column variants are supported but may need adaptation for unseen releases.
- Policy PDF parsing depends on optional pypdf.
- HTML parsing is best effort.
- Fake embeddings are not semantic.
- Milvus real mode requires external service availability.
- FaithfulnessProxy is not a substitute for expert review.
- This project is not legal advice.

## Stage completion status

Stages 0-8 are implemented for the MVP staged workflow:

- Stage 0: fixed scaffold
- Stage 1: schemas and core interfaces
- Stage 2: parsers and scripts 01-04
- Stage 3: chunking and script 05
- Stage 4: DuckDB and script 06
- Stage 5: Milvus, BM25, RRF, rerankers, HybridRetriever, and script 07
- Stage 6: Agentic RAG workflow and script 08
- Stage 7: evaluation, ablation, and scripts 09-10
- Stage 8: fixture-based end-to-end pipeline and final documentation

## Phase 0: Data ingestion and parser quality

Phase 0 hardens the ingestion/parsing layer before retrieval, reranking, LLM answering, or chat improvements. The current project version focuses on normalized evidence from:

- Trademark records
- Patent records and claim/long-text records
- Patent litigation records

Policy corpus ingestion may still exist for compatibility, but policy evidence is not required for default risk analysis in this version. Bad parsing creates incomplete documents, which then create bad chunks, weak retrieval, unreliable reranking, and poor LLM answers.

Example parser and quality-check commands:

```bash
python scripts/01_parse_trademark_xml.py --input data/raw/trademarks --output data/processed/trademarks.jsonl --report data/processed/trademark_report.json
python scripts/02_parse_patent_tsv.py --input data/raw/patents --output data/processed/patents.jsonl --report data/processed/patent_report.json
python scripts/03_parse_litigation_csv.py --input data/raw/litigation --output data/processed/litigation.jsonl --report data/processed/litigation_report.json
python scripts/check_ingestion_quality.py \
  --input data/processed/normalized_docs.jsonl \
  --require-source-types trademark,patent,litigation
```

This system supports compliance research and retrieval workflows, but it is not legal advice.


## Phase 1: Runtime CLI and Milvus Lite retrieval

Phase 1 makes runtime retrieval easier to run without pasted Python heredocs. It adds stable CLI entry points for LLM API smoke tests, dense Milvus retrieval, and BM25/dense/hybrid retrieval. The retrieval scripts prefer `RAG_MILVUS_URI` for local Milvus Lite databases and only fall back to `MILVUS_URI` for backward compatibility.

### Environment setup example

```bash
export RAG_MILVUS_URI=/root/autodl-tmp/Agentic_Rag/data/milvus_qa_300k.db
export LOCAL_EMBEDDING_MODEL=/root/autodl-tmp/models/bge-small-en-v1.5
export EMBEDDING_PROVIDER=local
export OPENAI_BASE_URL=https://api-inference.modelscope.cn/v1
export LLM_MODEL=deepseek-ai/DeepSeek-V4-Pro
```

Set `OPENAI_API_KEY` in your shell or secret manager before running the LLM smoke test. Do not commit `.env`, model weights, local database files, or generated output artifacts.

### Test LLM API

```bash
python scripts/test_llm_api.py
```

The script prints whether `OPENAI_API_KEY` is set, but it never prints the key. It also prints `OPENAI_BASE_URL`, `LLM_MODEL`, and the final model response. Empty `choices` or empty message content fail clearly.

### Dense query

```bash
python scripts/run_dense_query.py \
  --query "Find patent claims related to drone delivery control." \
  --top-k 10
```

Dense retrieval requires `RAG_MILVUS_URI` (or legacy `MILVUS_URI`), a collection such as `ip_chunks_qa_300k`, `pymilvus`, and a working embedding provider. Milvus Lite collections are loaded before search so a released collection does not fail with `call load() before search/get/query`.

### BM25-only query without Milvus

```bash
python scripts/run_hybrid_query.py \
  --query "smart travel bag trademark risk" \
  --mode bm25_only \
  --top-k 10
```

`bm25_only` reads the chunks JSONL and does not require Milvus, `pymilvus`, embeddings, `torch`, or `sentence-transformers`.

### Hybrid RRF query

```bash
python scripts/run_hybrid_query.py \
  --query "What trademark and patent risks should a seller consider for a smart travel bag product?" \
  --mode hybrid_rrf \
  --top-k 10 \
  --output-json
```

Hybrid modes combine local BM25 with dense Milvus results and include compact hit previews plus `source_type_counts` and `source_subtype_counts` in JSON output.

### Phase 2: Hybrid reranking

`hybrid_rerank` now uses a real two-stage candidate pool:

BM25 top `candidate_k` + Dense top `candidate_k` → RRF fusion top `candidate_k` → de-duplication → reranker → final top `top_k` evidence.

`candidate_k` controls the candidate pool before reranking. `top_k` controls the final evidence count passed to deterministic answer synthesis. The `lexical` reranker is dependency-free and useful for smoke tests. The `local` cross-encoder reranker requires `sentence-transformers` and a working `torch` installation. This system supports retrieval and compliance research, but it is still not legal advice.

Hybrid RRF baseline:

```bash
python scripts/run_hybrid_query.py \
  --query "What trademark and patent risks should a seller consider for a smart travel bag product?" \
  --mode hybrid_rrf \
  --top-k 8 \
  --output-json
```

Hybrid rerank with the lexical reranker:

```bash
python scripts/run_hybrid_query.py \
  --query "What trademark and patent risks should a seller consider for a smart travel bag product?" \
  --mode hybrid_rerank \
  --reranker-provider lexical \
  --candidate-k 50 \
  --top-k 8 \
  --output-json
```

Hybrid rerank with a local cross-encoder:

```bash
python scripts/run_hybrid_query.py \
  --query "What trademark and patent risks should a seller consider for a smart travel bag product?" \
  --mode hybrid_rerank \
  --reranker-provider local \
  --reranker-model BAAI/bge-reranker-base \
  --candidate-k 50 \
  --top-k 8 \
  --output-json
```

### Troubleshooting

* **`RAG_MILVUS_URI` missing**: set `export RAG_MILVUS_URI=/path/to/milvus.db`. `MILVUS_URI` is only a backward-compatible fallback.
* **Milvus Lite collection released**: the runtime store calls `load_collection(collection_name=...)` before search in Lite mode and `collection.load()` in server mode.
* **Malformed `metadata_json`**: retrieval maps empty metadata to `{}` and malformed metadata to `{ "_metadata_parse_error": true }` instead of crashing.
* **Optional `pymilvus` missing**: install `pymilvus` for dense/hybrid Milvus retrieval, or run `--mode bm25_only` without Milvus.
* **Optional `torch`/`sentence-transformers` DLL issue on Windows**: use `RERANKER_PROVIDER=noop` or `lexical` for baseline tests. Local cross-encoder reranking requires both `sentence-transformers` and a working `torch` installation.
* **LLM API returns empty choices**: verify `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `LLM_MODEL`, then rerun `python scripts/test_llm_api.py`.
* **Generated-file safety**: do not commit `data/`, `.env`, model weights, Milvus `.db` files, archives, or output artifacts.

### Legal disclaimer

This system supports retrieval and compliance research, but it is not legal advice.
