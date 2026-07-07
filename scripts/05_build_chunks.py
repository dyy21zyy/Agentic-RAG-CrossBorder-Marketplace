from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crossborder_agentic_rag.ingestion.chunkers import chunk_document
from crossborder_agentic_rag.ingestion.io_utils import read_documents_jsonl, write_chunks_jsonl, write_report


def _empty_report(input_path: str | Path, output_path: str | Path) -> dict:
    return {
        "input": str(input_path),
        "output": str(output_path),
        "documents_seen": 0,
        "documents_chunked": 0,
        "chunks_written": 0,
        "total_chunks": 0,
        "unique_chunk_ids": 0,
        "duplicate_chunk_ids": 0,
        "duplicate_chunk_id_examples": [],
        "chunks_by_source_type": {},
        "chunks_by_source_subtype": {},
        "failed_documents": [],
        "warnings": [],
    }


def add_duplicate_chunk_id_stats(report: dict, chunks: list) -> None:
    counts = Counter(chunk.chunk_id for chunk in chunks)
    duplicate_examples = [
        {"chunk_id": chunk_id, "count": count}
        for chunk_id, count in counts.most_common()
        if count > 1
    ][:20]
    report["total_chunks"] = len(chunks)
    report["unique_chunk_ids"] = len(counts)
    report["duplicate_chunk_ids"] = len(duplicate_examples)
    report["duplicate_chunk_id_examples"] = duplicate_examples
    if duplicate_examples:
        duplicate_extra = len(chunks) - len(counts)
        report["warnings"].append(
            f"Input generated duplicate chunk_id values: duplicate_id_count={len(duplicate_examples)}, duplicate_extra={duplicate_extra}."
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build EvidenceChunk JSONL from NormalizedDocument JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--fail-on-duplicate-chunk-id", action="store_true")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    report = _empty_report(input_path, args.output)
    if not input_path.is_file():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    try:
        docs = read_documents_jsonl(input_path)
    except Exception as exc:  # pragma: no cover - defensive CLI reporting
        print(f"Failed to read input JSONL: {exc}", file=sys.stderr)
        return 1

    report["documents_seen"] = len(docs)
    if not docs and not args.allow_empty:
        print("Input JSONL is empty; pass --allow-empty to write an empty chunk file.", file=sys.stderr)
        return 1
    if not docs:
        report["warnings"].append("Input JSONL was empty; wrote an empty chunk file because --allow-empty was passed.")

    chunks = []
    by_source_type: Counter[str] = Counter()
    by_subtype: Counter[str] = Counter()
    for doc in docs:
        try:
            doc_chunks = chunk_document(doc)
        except Exception as exc:  # continue on per-document chunking failures
            report["failed_documents"].append({"doc_id": doc.doc_id, "source_type": doc.source_type, "error": str(exc)})
            continue
        if doc_chunks:
            report["documents_chunked"] += 1
        chunks.extend(doc_chunks)
        by_source_type.update(chunk.source_type for chunk in doc_chunks)
        by_subtype.update(chunk.source_subtype for chunk in doc_chunks)

    add_duplicate_chunk_id_stats(report, chunks)
    if report["duplicate_chunk_ids"] and args.fail_on_duplicate_chunk_id:
        write_report(report, args.report)
        print(f"Duplicate chunk_id values detected; see report: {args.report}", file=sys.stderr)
        return 1
    report["chunks_written"] = write_chunks_jsonl(chunks, args.output)
    report["chunks_by_source_type"] = dict(sorted(by_source_type.items()))
    report["chunks_by_source_subtype"] = dict(sorted(by_subtype.items()))
    write_report(report, args.report)
    print(f"Chunked {report['documents_chunked']}/{report['documents_seen']} documents into {report['chunks_written']} chunks.")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
