"""Stage 0 scaffold tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import crossborder_agentic_rag

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "crossborder_agentic_rag"

TOP_LEVEL_FILES = {
    "README.md",
    "pyproject.toml",
    ".env.example",
    ".gitignore",
    "docker-compose.yml",
}

CONFIG_FILES = {
    "configs/paths.yaml",
    "configs/duckdb.yaml",
    "configs/milvus.yaml",
    "configs/retrieval.yaml",
    "configs/agents.yaml",
    "configs/evaluation.yaml",
}

PACKAGE_DIRECTORIES = {
    "config",
    "schemas",
    "ingestion",
    "storage",
    "retrieval",
    "llm",
    "agents",
    "evaluation",
    "utils",
}

MODULE_FILES = {
    "__init__.py",
    "constants.py",
    "config/__init__.py",
    "config/settings.py",
    "schemas/__init__.py",
    "schemas/documents.py",
    "schemas/evidence.py",
    "schemas/queries.py",
    "schemas/results.py",
    "ingestion/__init__.py",
    "ingestion/io_utils.py",
    "ingestion/trademark_parser.py",
    "ingestion/patent_parser.py",
    "ingestion/litigation_parser.py",
    "ingestion/policy_parser.py",
    "ingestion/chunkers.py",
    "storage/__init__.py",
    "storage/duckdb_store.py",
    "storage/milvus_store.py",
    "storage/schemas.py",
    "retrieval/__init__.py",
    "retrieval/bm25.py",
    "retrieval/hybrid_retriever.py",
    "retrieval/rrf_fusion.py",
    "retrieval/reranker.py",
    "llm/__init__.py",
    "llm/embeddings.py",
    "llm/generation.py",
    "agents/__init__.py",
    "agents/classify.py",
    "agents/planner.py",
    "agents/sql_router.py",
    "agents/evaluator.py",
    "agents/answer.py",
    "agents/graph.py",
    "evaluation/__init__.py",
    "evaluation/metrics.py",
    "evaluation/datasets.py",
    "evaluation/evaluator.py",
    "evaluation/ablations.py",
    "evaluation/report.py",
    "utils/__init__.py",
    "utils/jsonl.py",
    "utils/text.py",
    "utils/logging.py",
}

SCRIPT_FILES = {
    "01_parse_trademark_xml.py",
    "02_parse_patent_tsv.py",
    "03_parse_litigation_csv.py",
    "04_parse_policy_docs.py",
    "05_build_chunks.py",
    "06_build_duckdb.py",
    "07_build_milvus_index.py",
    "08_run_query_cli.py",
    "09_run_eval.py",
    "10_run_ablation.py",
}

ENV_KEYS = {
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_API_BASE",
    "LLM_MODEL",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_API_KEY",
    "EMBEDDING_API_BASE",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "RERANKER_PROVIDER",
    "RERANKER_API_KEY",
    "RERANKER_API_BASE",
    "RERANKER_MODEL",
    "MILVUS_URI",
    "MILVUS_TOKEN",
    "MILVUS_COLLECTION_NAME",
    "DUCKDB_PATH",
    "TRADEMARK_RAW_DIR",
    "PATENT_RAW_DIR",
    "LITIGATION_RAW_DIR",
    "POLICY_RAW_DIR",
    "MAX_RETRIEVAL_ITERATIONS",
}


def test_package_imports() -> None:
    assert crossborder_agentic_rag.__name__ == "crossborder_agentic_rag"


def test_version_exists() -> None:
    assert crossborder_agentic_rag.__version__ == "0.1.0"


def test_required_top_level_files_exist() -> None:
    assert all((ROOT / path).is_file() for path in TOP_LEVEL_FILES)


def test_required_config_files_exist() -> None:
    assert all((ROOT / path).is_file() for path in CONFIG_FILES)


def test_required_package_directories_exist() -> None:
    assert all((PACKAGE_ROOT / path).is_dir() for path in PACKAGE_DIRECTORIES)


def test_required_module_files_exist() -> None:
    assert all((PACKAGE_ROOT / path).is_file() for path in MODULE_FILES)


def test_required_script_files_exist() -> None:
    assert all((ROOT / "scripts" / path).is_file() for path in SCRIPT_FILES)


def test_env_example_contains_required_keys() -> None:
    content = (ROOT / ".env.example").read_text(encoding="utf-8")
    present_keys = {line.split("=", 1)[0] for line in content.splitlines() if "=" in line}
    assert ENV_KEYS <= present_keys


def test_gitignore_excludes_local_data_and_secrets() -> None:
    content = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    required_patterns = {".env", ".env.*", "!.env.example", "data/raw/", "data/processed/", "data/eval/", "*.duckdb"}
    assert required_patterns <= set(content)


def test_scripts_fail_with_not_implemented() -> None:
    """All scaffold scripts through Stage 7 are implemented."""
    for script_name in SCRIPT_FILES:
        text = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert "NotImplementedError" not in text
