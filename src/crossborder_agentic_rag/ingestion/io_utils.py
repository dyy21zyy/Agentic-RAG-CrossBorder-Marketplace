"""Ingestion I/O utilities for Stage 2 parsers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from crossborder_agentic_rag.schemas.documents import NormalizedDocument


def ensure_input_dir(input_dir: str | Path) -> Path:
    path = Path(input_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {path}")
    return path


def ensure_parent_dir(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def write_documents_jsonl(docs: Iterable[NormalizedDocument], output_path: str | Path) -> int:
    path = ensure_parent_dir(output_path)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_documents_jsonl(input_path: str | Path) -> list[NormalizedDocument]:
    docs: list[NormalizedDocument] = []
    with Path(input_path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                docs.append(NormalizedDocument.from_dict(json.loads(line)))
    return docs


def write_report(report: dict, report_path: str | Path) -> None:
    path = ensure_parent_dir(report_path)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def init_report(source_type: str, input_dir: str | Path) -> dict:
    return {
        "source_type": source_type,
        "input_dir": str(Path(input_dir)),
        "files_seen": 0,
        "files_parsed": 0,
        "documents_parsed": 0,
        "failed_files": [],
        "warnings": [],
    }
