"""Patent TSV parser."""
from __future__ import annotations
import csv
from pathlib import Path
from crossborder_agentic_rag.ingestion.io_utils import ensure_input_dir, init_report
from crossborder_agentic_rag.schemas.documents import NormalizedDocument

VARIANTS = {
    "patent_id": ["patent_id", "patent_number", "id", "pat_no"],
    "brief_summary": ["brief_summary", "Brief Summary", "summary"],
    "claims": ["claims", "Claims", "claim_txt"],
    "detail_description": ["detail_description", "Detail Description", "description"],
    "drawing_description": ["drawing_description", "Drawing Description", "drawings"],
}

def _get(row: dict, names: list[str]) -> str:
    low = {str(k).lower().replace(" ", "_"): v for k, v in row.items()}
    for name in names:
        val = row.get(name)
        if val is None:
            val = low.get(name.lower().replace(" ", "_"))
        if val and str(val).strip(): return str(val).strip()
    return ""

def parse_patent_tsv_directory(input_dir: str | Path) -> tuple[list[NormalizedDocument], dict]:
    base = ensure_input_dir(input_dir); report = init_report("patent", base); docs=[]
    for path in sorted(base.rglob("*.tsv")):
        report["files_seen"] += 1
        try:
            with path.open("r", encoding="utf-8", newline="") as fh:
                for n, row in enumerate(csv.DictReader(fh, delimiter="\t"), start=2):
                    patent_id = _get(row, VARIANTS["patent_id"])
                    if not patent_id:
                        report["warnings"].append({"path": str(path), "row": n, "warning": "skipped row without patent id"}); continue
                    md = {k: _get(row, v) for k, v in VARIANTS.items()}
                    md.update({"source_file": path.name, "source_path": str(path)})
                    content = "\n".join(f"{label}: {md[key]}" for key, label in [("brief_summary","Brief summary"),("claims","Claims"),("detail_description","Detail description"),("drawing_description","Drawing description")] if md[key])
                    docs.append(NormalizedDocument(f"patent:{patent_id}", "patent", patent_id, content, md))
            report["files_parsed"] += 1
        except Exception as exc:
            report["failed_files"].append({"path": str(path), "error": str(exc)})
    report["documents_parsed"] = len(docs); return docs, report
