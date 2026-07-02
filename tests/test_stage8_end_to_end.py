from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "e2e"


def run_cmd(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise AssertionError(f"Command failed: {' '.join(args)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_end_to_end_fixture_pipeline(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    eval_dir = tmp_path / "eval"
    ablation_dir = tmp_path / "ablation"
    processed.mkdir()

    commands = [
        [sys.executable, "scripts/01_parse_trademark_xml.py", "--input", str(FIXTURES / "trademark"), "--output", str(processed / "trademarks.jsonl"), "--report", str(processed / "trademark_report.json")],
        [sys.executable, "scripts/02_parse_patent_tsv.py", "--input", str(FIXTURES / "patent"), "--output", str(processed / "patents.jsonl"), "--report", str(processed / "patent_report.json")],
        [sys.executable, "scripts/03_parse_litigation_csv.py", "--input", str(FIXTURES / "litigation"), "--output", str(processed / "litigation.jsonl"), "--report", str(processed / "litigation_report.json")],
        [sys.executable, "scripts/04_parse_policy_docs.py", "--input", str(FIXTURES / "policies"), "--output", str(processed / "policies.jsonl"), "--report", str(processed / "policy_report.json")],
    ]
    for command in commands:
        run_cmd(command)

    all_docs = processed / "all_docs.jsonl"
    with all_docs.open("w", encoding="utf-8") as out:
        for name in ["trademarks.jsonl", "patents.jsonl", "litigation.jsonl", "policies.jsonl"]:
            out.write((processed / name).read_text(encoding="utf-8"))

    run_cmd([sys.executable, "scripts/05_build_chunks.py", "--input", str(all_docs), "--output", str(processed / "chunks.jsonl"), "--report", str(processed / "chunk_report.json")])
    run_cmd([sys.executable, "scripts/06_build_duckdb.py", "--input", str(all_docs), "--duckdb-path", str(processed / "ip.duckdb"), "--report", str(processed / "duckdb_report.json"), "--overwrite"])
    run_cmd([sys.executable, "scripts/07_build_milvus_index.py", "--input", str(processed / "chunks.jsonl"), "--dry-run", "--report", str(processed / "milvus_report.json")])

    policy = run_cmd([sys.executable, "scripts/08_run_query_cli.py", "Explain trademark infringement", "--duckdb-path", str(processed / "ip.duckdb"), "--chunks-path", str(processed / "chunks.jsonl"), "--output-json"])
    policy_json = json.loads(policy.stdout)
    assert policy_json["answer"]
    assert policy_json["retrieval_route"]
    assert "citations" in policy_json
    assert "trace" in policy_json
    assert "Risk Level" not in policy_json["answer"]

    risk = run_cmd([sys.executable, "scripts/08_run_query_cli.py", "Can I sell a phone case using the MERCEDES logo?", "--duckdb-path", str(processed / "ip.duckdb"), "--chunks-path", str(processed / "chunks.jsonl"), "--output-json"])
    risk_json = json.loads(risk.stdout)
    assert "Risk Level" in risk_json["answer"]
    assert risk_json["retrieval_route"] == "multi_source_risk" or risk_json["query_type"] == "risk_analysis"

    eval_file = FIXTURES / "eval" / "eval_queries.jsonl"
    run_cmd([sys.executable, "scripts/09_run_eval.py", "--eval-file", str(eval_file), "--output-dir", str(eval_dir), "--duckdb-path", str(processed / "ip.duckdb"), "--chunks-path", str(processed / "chunks.jsonl")])
    for name in ["eval_results.json", "eval_results.csv", "eval_summary.json", "eval_summary.md"]:
        assert (eval_dir / name).is_file()

    run_cmd([sys.executable, "scripts/10_run_ablation.py", "--eval-file", str(eval_file), "--output-dir", str(ablation_dir), "--duckdb-path", str(processed / "ip.duckdb"), "--chunks-path", str(processed / "chunks.jsonl"), "--experiments", "bm25_only,hybrid_rrf,no_reranker"])
    for name in ["ablation_results.json", "ablation_results.csv", "ablation_summary.md"]:
        assert (ablation_dir / name).is_file()

    assert load_json(processed / "trademark_report.json")["documents_parsed"] >= 1
    assert load_json(processed / "patent_report.json")["documents_parsed"] >= 1
    assert load_json(processed / "litigation_report.json")["documents_parsed"] >= 1
    assert load_json(processed / "policy_report.json")["documents_parsed"] == 0
    assert load_json(processed / "chunk_report.json")["chunks_written"] >= 1
    row_counts = load_json(processed / "duckdb_report.json")["row_counts"]
    assert row_counts.get("trademarks", 0) >= 1
    assert row_counts.get("patents", 0) >= 1
    assert row_counts.get("litigation_cases", 0) >= 1
    milvus_report = load_json(processed / "milvus_report.json")
    assert milvus_report["dry_run"] is True
    assert milvus_report["milvus_inserted"] == 0


def test_readme_contains_full_pipeline_commands() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for needle in [
        "scripts/01_parse_trademark_xml.py", "scripts/02_parse_patent_tsv.py", "scripts/03_parse_litigation_csv.py", "scripts/04_parse_policy_docs.py", "scripts/05_build_chunks.py", "scripts/06_build_duckdb.py", "scripts/07_build_milvus_index.py", "scripts/08_run_query_cli.py", "scripts/09_run_eval.py", "scripts/10_run_ablation.py", "docker compose up -d", "FakeEmbeddingProvider is only for tests and smoke runs", "Real Milvus mode requires a running Milvus instance", "FaithfulnessProxy is a heuristic", "Only risk_analysis answers include Risk Level",
    ]:
        assert needle in readme


def test_no_production_script_uses_empty_retriever_by_default(tmp_path: Path) -> None:
    query = run_cmd([sys.executable, "scripts/08_run_query_cli.py", "Explain trademark infringement"], check=False)
    assert query.returncode != 0
    assert any(term in (query.stdout + query.stderr).lower() for term in ["retrieval backend", "chunks path", "milvus", "demo mode"])
    eval_result = run_cmd([sys.executable, "scripts/09_run_eval.py", "--eval-file", str(FIXTURES / "eval" / "eval_queries.jsonl"), "--output-dir", str(tmp_path / "eval_no_backend")], check=False)
    assert eval_result.returncode != 0
    assert any(term in (eval_result.stdout + eval_result.stderr).lower() for term in ["backend", "demo mode", "chunks-path", "milvus"])


def test_empty_input_fails_without_allow_empty(tmp_path: Path) -> None:
    empty_docs = tmp_path / "empty_docs.jsonl"
    empty_chunks = tmp_path / "empty_chunks.jsonl"
    empty_docs.write_text("", encoding="utf-8")
    empty_chunks.write_text("", encoding="utf-8")
    checks = [
        [sys.executable, "scripts/05_build_chunks.py", "--input", str(empty_docs), "--output", str(tmp_path / "chunks.jsonl"), "--report", str(tmp_path / "chunk_report.json")],
        [sys.executable, "scripts/06_build_duckdb.py", "--input", str(empty_docs), "--duckdb-path", str(tmp_path / "ip.duckdb"), "--report", str(tmp_path / "duckdb_report.json")],
        [sys.executable, "scripts/07_build_milvus_index.py", "--input", str(empty_chunks), "--dry-run", "--report", str(tmp_path / "milvus_report.json")],
    ]
    for command in checks:
        result = run_cmd(command, check=False)
        assert result.returncode != 0
        assert "empty" in (result.stdout + result.stderr).lower()


def test_no_duplicate_module_paths_created() -> None:
    for path in [
        "src/crossborder_agentic_rag/e2e/", "src/crossborder_agentic_rag/pipeline/", "src/crossborder_agentic_rag/runner/", "src/crossborder_agentic_rag/eval/", "src/crossborder_agentic_rag/benchmark/", "src/crossborder_agentic_rag/vectorstore/", "src/crossborder_agentic_rag/retriever/",
    ]:
        assert not (ROOT / path).exists()
