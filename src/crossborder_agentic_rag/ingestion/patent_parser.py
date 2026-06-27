"""Patent TSV parser."""
from __future__ import annotations
import csv
from pathlib import Path
from crossborder_agentic_rag.ingestion.io_utils import ensure_input_dir, init_report
from crossborder_agentic_rag.schemas.documents import NormalizedDocument

VARIANTS = {
    "patent_id": ["patent_id", "patent_number", "publication_number", "document_number", "id", "patent", "pat_no"],
    "title": ["title", "invention_title", "patent_title"],
    "abstract": ["abstract"],
    "brief_summary": ["brief_summary", "summary", "brief_summary_text", "Brief Summary"],
    "claims": ["claim_text", "claims", "claim", "text", "claim_fulltext", "claim_text_full", "Claims", "claim_txt"],
    "detail_description": ["detail_description", "detailed_description", "description", "detail_desc_text", "Detail Description"],
    "drawing_description": ["drawing_description", "drawing_desc_text", "Drawing Description", "drawings"],
    "patent_date": ["patent_date", "grant_date", "publication_date", "date"],
    "claim_number": ["claim_number", "claim_num", "claim_sequence", "sequence"],
    "claim_type": ["claim_type", "type"],
    "is_independent": ["is_independent", "independent", "dependent"],
}

SUBTYPES = [
    ("patent_abstract", "abstract", "Abstract"),
    ("patent_summary", "brief_summary", "Brief summary"),
    ("patent_claim", "claims", "Claim"),
    ("patent_description", "detail_description", "Detailed description"),
    ("patent_drawing_description", "drawing_description", "Drawing description"),
]


def _key(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def _get(row: dict, names: list[str]) -> str:
    low = {_key(k): v for k, v in row.items()}
    for name in names:
        val = row.get(name)
        if val is None:
            val = low.get(_key(name))
        if val and str(val).strip():
            return str(val).strip()
    return ""


def _subtype_for(md: dict) -> str:
    if md.get("claims"): return "patent_claim"
    if md.get("abstract"): return "patent_abstract"
    if md.get("brief_summary"): return "patent_summary"
    if md.get("detail_description"): return "patent_description"
    if md.get("drawing_description"): return "patent_drawing_description"
    return "patent_metadata"


def parse_patent_tsv_directory(input_dir: str | Path) -> tuple[list[NormalizedDocument], dict]:
    base = ensure_input_dir(input_dir); report = init_report("patent", base); docs=[]; seen=set()
    for path in sorted(base.rglob("*.tsv")):
        report["files_seen"] += 1
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                for n, row in enumerate(csv.DictReader(fh, delimiter="\t"), start=2):
                    patent_id = _get(row, VARIANTS["patent_id"])
                    if not patent_id:
                        report["warnings"].append({"path": str(path), "row": n, "warning": "skipped row without patent id"}); continue
                    md = {k: _get(row, v) for k, v in VARIANTS.items()}
                    md.update({"patent_id": patent_id, "source_file": path.name, "source_path": str(path)})
                    labels = [("abstract","Abstract"),("brief_summary","Brief summary"),("claims","Claims"),("detail_description","Detailed description"),("drawing_description","Drawing description")]
                    content = "\n".join(f"{label}: {md[key]}" for key, label in labels if md.get(key))
                    if not content:
                        if md.get("title") or md.get("patent_date"):
                            content = "\n".join(f"{label}: {value}" for label, value in [("Patent ID", patent_id), ("Title", md.get("title")), ("Date", md.get("patent_date"))] if value)
                        else:
                            report["warnings"].append({"path": str(path), "row": n, "patent_id": patent_id, "warning": "skipped row without text fields"}); continue
                    if md.get("claims"):
                        content = f"Patent ID: {patent_id}\n" + content
                    md["source_subtype"] = _subtype_for(md)
                    indep = md.get("is_independent", "").lower()
                    if indep in {"true", "1", "yes", "independent"}: md["is_independent"] = True
                    elif indep in {"false", "0", "no", "dependent"}: md["is_independent"] = False
                    doc_id = f"patent:{patent_id}"
                    if doc_id in seen:
                        report["warnings"].append({"path": str(path), "row": n, "doc_id": doc_id, "warning": "duplicate doc_id skipped"}); continue
                    seen.add(doc_id)
                    docs.append(NormalizedDocument(doc_id, "patent", md.get("title") or patent_id, content, md))
            report["files_parsed"] += 1
        except Exception as exc:
            report["failed_files"].append({"path": str(path), "error": str(exc)})
    report["documents_parsed"] = len(docs); return docs, report
