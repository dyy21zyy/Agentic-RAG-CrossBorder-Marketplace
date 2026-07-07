from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crossborder_agentic_rag.ingestion.chunkers import chunk_document
from crossborder_agentic_rag.ingestion.io_utils import ensure_parent_dir, write_report
from crossborder_agentic_rag.schemas.documents import NormalizedDocument


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
        "failed_documents_count": 0,
        "failed_documents": [],
        "chunks_by_source_type": {},
        "chunks_by_source_subtype": {},
        "average_chunks_per_document": 0.0,
        "approximate_output_size_bytes": 0,
        "approximate_output_size_mb": 0.0,
        "average_bytes_per_chunk": 0.0,
        "average_bytes_per_document": 0.0,
        "warnings": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stream EvidenceChunk JSONL from NormalizedDocument JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--fail-on-duplicate-chunk-id", action="store_true")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = ensure_parent_dir(args.output)
    report = _empty_report(input_path, output_path)
    if not input_path.is_file():
        print(f"Input file does not exist: {input_path}", file=sys.stderr)
        return 1

    by_source_type: Counter[str] = Counter()
    by_subtype: Counter[str] = Counter()
    chunk_id_counts: Counter[str] = Counter()
    failed: list[dict] = []
    total_bytes = 0

    with input_path.open("r", encoding="utf-8") as in_fh, output_path.open("w", encoding="utf-8") as out_fh:
        for line_no, line in enumerate(in_fh, 1):
            if not line.strip():
                continue
            try:
                doc = NormalizedDocument.from_dict(json.loads(line))
                report["documents_seen"] += 1
                doc_chunks = chunk_document(doc)
                if doc_chunks:
                    report["documents_chunked"] += 1
                for chunk in doc_chunks:
                    payload = json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n"
                    out_fh.write(payload)
                    total_bytes += len(payload.encode("utf-8"))
                    report["chunks_written"] += 1
                    chunk_id_counts[chunk.chunk_id] += 1
                    by_source_type[chunk.source_type] += 1
                    by_subtype[chunk.source_subtype] += 1
            except Exception as exc:  # pragma: no cover - defensive CLI reporting
                report["failed_documents_count"] += 1
                if len(failed) < 1000:
                    failed.append({"line": line_no, "error": str(exc)})
            if args.progress_every > 0 and report["documents_seen"] and report["documents_seen"] % args.progress_every == 0:
                print(f"Processed {report['documents_seen']} documents; wrote {report['chunks_written']} chunks.", file=sys.stderr)

    if report["documents_seen"] == 0 and not args.allow_empty:
        print("Input JSONL is empty; pass --allow-empty to write an empty chunk file.", file=sys.stderr)
        return 1
    if report["documents_seen"] == 0:
        report["warnings"].append("Input JSONL was empty; wrote an empty chunk file because --allow-empty was passed.")

    report["failed_documents"] = failed
    report["chunks_by_source_type"] = dict(sorted(by_source_type.items()))
    report["chunks_by_source_subtype"] = dict(sorted(by_subtype.items()))
    duplicate_examples = [{"chunk_id": cid, "count": count} for cid, count in chunk_id_counts.most_common() if count > 1][:20]
    report["total_chunks"] = report["chunks_written"]
    report["unique_chunk_ids"] = len(chunk_id_counts)
    report["duplicate_chunk_ids"] = len(duplicate_examples)
    report["duplicate_chunk_id_examples"] = duplicate_examples
    if duplicate_examples:
        report["warnings"].append(
            f"Input generated duplicate chunk_id values: duplicate_id_count={len(duplicate_examples)}, duplicate_extra={report['chunks_written'] - len(chunk_id_counts)}."
        )
    docs_seen = report["documents_seen"] or 0
    chunks_written = report["chunks_written"] or 0
    report["average_chunks_per_document"] = chunks_written / docs_seen if docs_seen else 0.0
    report["approximate_output_size_bytes"] = total_bytes
    report["approximate_output_size_mb"] = round(total_bytes / (1024 * 1024), 3)
    report["average_bytes_per_chunk"] = total_bytes / chunks_written if chunks_written else 0.0
    report["average_bytes_per_document"] = total_bytes / docs_seen if docs_seen else 0.0
    write_report(report, args.report)
    if report["duplicate_chunk_ids"] and args.fail_on_duplicate_chunk_id:
        print(f"Duplicate chunk_id values detected; see report: {args.report}", file=sys.stderr)
        return 1
    print(f"Chunked {report['documents_chunked']}/{report['documents_seen']} documents into {report['chunks_written']} chunks.")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
