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
from crossborder_agentic_rag.schemas.documents import NormalizedDocument


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile optimized chunk JSONL size for a sample input.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--total-docs", type=int)
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    docs = chunks = size = 0
    by_source_type: Counter[str] = Counter()
    by_subtype: Counter[str] = Counter()
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            doc = NormalizedDocument.from_dict(json.loads(line))
            docs += 1
            for chunk in chunk_document(doc):
                payload = json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n"
                size += len(payload.encode("utf-8"))
                chunks += 1
                by_source_type[chunk.source_type] += 1
                by_subtype[chunk.source_subtype] += 1
    report = {
        "input": str(input_path),
        "docs_count": docs,
        "chunks_count": chunks,
        "file_size": size,
        "file_size_mb": round(size / (1024 * 1024), 3),
        "chunks_by_source_type": dict(sorted(by_source_type.items())),
        "chunks_by_source_subtype": dict(sorted(by_subtype.items())),
    }
    if args.total_docs and docs:
        report["estimated_full_size_bytes"] = int(size * (args.total_docs / docs))
        report["estimated_full_size_mb"] = round(report["estimated_full_size_bytes"] / (1024 * 1024), 3)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
