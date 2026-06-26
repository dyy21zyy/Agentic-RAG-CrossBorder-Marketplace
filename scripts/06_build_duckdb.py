from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crossborder_agentic_rag.ingestion.io_utils import read_documents_jsonl, write_report
from crossborder_agentic_rag.storage.duckdb_store import DuckDBStore


def _base_report(input_path: str | Path, duckdb_path: str | Path) -> dict:
    return {
        "input": str(input_path),
        "duckdb_path": str(duckdb_path),
        "documents_seen": 0,
        "documents_loaded": 0,
        "documents_skipped": 0,
        "rows_inserted": {},
        "row_counts": {},
        "failed_documents": [],
        "warnings": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a DuckDB structured lookup database from NormalizedDocument JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--duckdb-path", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    duckdb_path = Path(args.duckdb_path)
    report = _base_report(input_path, duckdb_path)

    if not input_path.is_file():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        return 1
    if args.overwrite and duckdb_path.exists():
        duckdb_path.unlink()
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    try:
        docs = read_documents_jsonl(input_path)
    except Exception as exc:
        print(f"Failed to read input JSONL: {exc}", file=sys.stderr)
        return 1

    report["documents_seen"] = len(docs)
    if not docs and not args.allow_empty:
        print("Input JSONL is empty; pass --allow-empty to create an empty DuckDB database.", file=sys.stderr)
        return 1
    if not docs:
        report["warnings"].append("Input JSONL was empty; created schema because --allow-empty was passed.")

    store = DuckDBStore(duckdb_path)
    try:
        store.initialize_schema()
        load_report = store.load_documents(docs) if docs else {"documents_loaded": 0, "documents_skipped": 0, "rows_inserted": {}, "failed_documents": [], "warnings": report["warnings"]}
        prior_warnings = list(report["warnings"])
        report.update(load_report)
        if prior_warnings:
            report["warnings"] = prior_warnings + [w for w in load_report.get("warnings", []) if w not in prior_warnings]
        report["input"] = str(input_path)
        report["duckdb_path"] = str(duckdb_path)
        report["row_counts"] = store.row_counts()
    except Exception as exc:
        print(f"Failed to build DuckDB database: {exc}", file=sys.stderr)
        report["warnings"].append(str(exc))
        write_report(report, args.report)
        return 1
    finally:
        store.close()

    write_report(report, args.report)
    if docs and report["documents_loaded"] == 0 and not args.allow_empty:
        print("All documents failed to load.", file=sys.stderr)
        return 1
    print(f"Loaded {report['documents_loaded']}/{report['documents_seen']} documents into {duckdb_path}.")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
