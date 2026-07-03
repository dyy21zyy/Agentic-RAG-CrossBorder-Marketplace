from __future__ import annotations

import ast
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [f"{i:02d}_" for i in range(1, 11)]
SCRIPT_FILES = sorted((ROOT / "scripts").glob("[0-9][0-9]_*.py"))


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def run_cmd(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True)


def test_readme_commands_reference_existing_scripts() -> None:
    readme = read("README.md")
    referenced = set(re.findall(r"python\s+(scripts/[0-9]{2}_[\w_]+\.py)", readme))
    assert len(referenced) >= 10
    missing = [p for p in referenced if not (ROOT / p).is_file()]
    assert missing == []


def test_readme_does_not_claim_legal_advice() -> None:
    text = read("README.md").lower()
    assert "not legal advice" in text
    assert "legal advice engine" not in text


def test_readme_mentions_fake_embedding_limitations() -> None:
    text = read("README.md")
    assert "FakeEmbeddingProvider is only for tests and smoke runs" in text
    assert "Real semantic retrieval requires" in text


def test_readme_mentions_milvus_real_mode_requirements() -> None:
    text = read("README.md")
    assert "Real Milvus mode requires a running Milvus instance" in text
    assert "pymilvus installed" in text
    assert "real embeddings" in text.lower()


def test_readme_mentions_dry_run_is_not_real_insertion() -> None:
    text = read("README.md")
    assert "Dry-run mode does not insert into Milvus" in text
    assert "not be interpreted as successful vector indexing" in text


def test_readme_mentions_faithfulness_proxy_is_heuristic() -> None:
    assert "FaithfulnessProxy is a heuristic" in read("README.md")


def test_all_scripts_have_main() -> None:
    assert len(SCRIPT_FILES) == 11
    for path in SCRIPT_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body), path


def test_scripts_01_to_10_do_not_raise_not_implemented_for_help() -> None:
    for path in SCRIPT_FILES:
        cp = run_cmd([sys.executable, str(path.relative_to(ROOT)), "--help"])
        assert cp.returncode == 0, (path.name, cp.stderr)
        assert "NotImplementedError" not in cp.stderr + cp.stdout


def test_query_cli_without_backend_fails_clearly() -> None:
    cp = run_cmd([sys.executable, "scripts/08_run_query_cli.py", "Explain trademark infringement"])
    assert cp.returncode != 0
    assert "No retrieval backend" in cp.stderr


def test_eval_cli_without_backend_fails_clearly(tmp_path: Path) -> None:
    cp = run_cmd([sys.executable, "scripts/09_run_eval.py", "--eval-file", "tests/fixtures/eval/eval_queries.jsonl", "--output-dir", str(tmp_path)])
    assert cp.returncode != 0
    assert "Normal mode requires --chunks-path or --use-milvus" in cp.stderr


def test_ablation_cli_without_backend_fails_clearly(tmp_path: Path) -> None:
    cp = run_cmd([sys.executable, "scripts/10_run_ablation.py", "--eval-file", "tests/fixtures/eval/eval_queries.jsonl", "--output-dir", str(tmp_path)])
    assert cp.returncode != 0
    assert "Normal mode requires --chunks-path or --use-milvus" in cp.stderr


def test_milvus_real_mode_without_service_fails_clearly(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.pop("EMBEDDING_PROVIDER", None)
    cp = subprocess.run([
        sys.executable, "scripts/07_build_milvus_index.py", "--input", "tests/fixtures/retrieval/sample_chunks.jsonl", "--report", str(tmp_path / "r.json")
    ], cwd=ROOT, text=True, capture_output=True, env=env)
    assert cp.returncode != 0
    assert "Real Milvus mode requires" in cp.stderr
    assert "FakeEmbeddingProvider" in cp.stderr


def test_milvus_dry_run_does_not_claim_insertion(tmp_path: Path) -> None:
    report = tmp_path / "milvus_report.json"
    cp = run_cmd([sys.executable, "scripts/07_build_milvus_index.py", "--input", "tests/fixtures/retrieval/sample_chunks.jsonl", "--dry-run", "--report", str(report)])
    assert cp.returncode == 0, cp.stderr
    assert "Milvus not contacted" in cp.stdout
    assert "Inserted" not in cp.stdout


def test_gitignore_generated_artifacts() -> None:
    lines = set(read(".gitignore").splitlines())
    for pattern in ["data/raw/", "data/processed/", "data/eval/", "*.duckdb", "*.db", "*.sqlite", ".env", ".env.*", "!.env.example", "__pycache__/", ".pytest_cache/"]:
        assert pattern in lines


def test_no_generated_files_committed_under_data() -> None:
    data_dir = ROOT / "data"
    if data_dir.exists():
        allowed_templates = {ROOT / "data/eval/golden_queries_30.jsonl"}
        assert [
            p
            for p in data_dir.rglob("*")
            if p.is_file() and p not in allowed_templates
        ] == []


def test_pyproject_has_optional_milvus_extra() -> None:
    text = read("pyproject.toml")
    assert "milvus =" in text and "pymilvus" in text


def test_pyproject_has_optional_local_extra() -> None:
    text = read("pyproject.toml")
    assert "local =" in text and "sentence-transformers" in text


def test_pyproject_has_duckdb_dependency() -> None:
    assert '"duckdb>=' in read("pyproject.toml")


def test_package_imports_cleanly() -> None:
    import crossborder_agentic_rag  # noqa: F401


def test_public_exports_available() -> None:
    from crossborder_agentic_rag import schemas
    from crossborder_agentic_rag.llm import BaseEmbeddingProvider, FakeEmbeddingProvider, build_embedding_provider
    assert schemas is not None
    assert BaseEmbeddingProvider and FakeEmbeddingProvider and build_embedding_provider


def test_no_duplicate_forbidden_paths() -> None:
    forbidden = [
        "src/crossborder_agentic_rag/e2e/", "src/crossborder_agentic_rag/pipeline/", "src/crossborder_agentic_rag/runner/",
        "src/crossborder_agentic_rag/eval/", "src/crossborder_agentic_rag/benchmark/", "src/crossborder_agentic_rag/vectorstore/",
        "src/crossborder_agentic_rag/retriever/", "src/crossborder_agentic_rag/storage/duckdb.py", "src/crossborder_agentic_rag/storage/milvus.py",
        "src/crossborder_agentic_rag/retrieval/hybrid.py",
    ]
    assert [p for p in forbidden if (ROOT / p).exists()] == []
