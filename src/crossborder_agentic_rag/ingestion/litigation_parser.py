"""Patent litigation CSV parser."""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from crossborder_agentic_rag.ingestion.io_utils import ensure_input_dir, init_report
from crossborder_agentic_rag.schemas.documents import NormalizedDocument

TABLES = ("cases", "documents", "names", "patents")

def litigation_case_key(row: dict) -> tuple[str, str, str]:
    return (str(row.get("case_row_id", "")).strip(), str(row.get("case_number", "")).strip(), str(row.get("district_id", "")).strip())

def _kind(path: Path) -> str | None:
    n = path.name.lower()
    return next((t for t in TABLES if t in n), None)

def _read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]

def _timeline(case, docs):
    events=[]
    if case.get("date_filed"): events.append({"date": case["date_filed"], "event_type": "filed", "description": "Case filed"})
    if case.get("date_closed"): events.append({"date": case["date_closed"], "event_type": "closed", "description": "Case closed"})
    for d in docs:
        if d.get("doc_date_filed"): events.append({"date": d["doc_date_filed"], "event_type": "docket", "description": d.get("short_description", "")})
    return sorted(events, key=lambda e: e.get("date") or "9999")

def parse_litigation_csv_directory(input_dir: str | Path) -> tuple[list[NormalizedDocument], dict]:
    base=ensure_input_dir(input_dir); report=init_report("litigation", base); rows={t: [] for t in TABLES}; files={t: [] for t in TABLES}; docs=[]
    for path in sorted(base.rglob("*.csv")):
        kind=_kind(path)
        if not kind: continue
        report["files_seen"] += 1
        try:
            rows[kind].extend(_read(path)); files[kind].append(path.name); report["files_parsed"] += 1
        except Exception as exc:
            report["failed_files"].append({"path": str(path), "error": str(exc)})
    if not rows["cases"]:
        report["warnings"].append({"warning": "no cases CSV rows found"})
    for t in ("documents", "names", "patents"):
        if not rows[t]: report["warnings"].append({"warning": f"missing related table: {t}"})
    grouped={t: defaultdict(list) for t in ("documents", "names", "patents")}
    for t in grouped:
        for r in rows[t]: grouped[t][litigation_case_key(r)].append(r)
    for case in rows["cases"]:
        key=litigation_case_key(case); ds=grouped["documents"].get(key, []); ps=grouped["patents"].get(key, []); ns=grouped["names"].get(key, [])
        source_files=sorted({f for t in TABLES for f in files[t]})
        patent_vals=[p.get("patent") or p.get("patent_number", "") for p in ps if p.get("patent") or p.get("patent_number")]
        parties=[n.get("name_long") or n.get("name", "") for n in ns if n.get("name_long") or n.get("name")]
        docket=[d.get("short_description", "") for d in ds if d.get("short_description")]
        content="\n".join(filter(None,[f"Case number: {case.get('case_number','')}", f"Case name: {case.get('case_name','')}", f"Court/district: {case.get('court_name','')} {case.get('district_id','')}", f"Date filed: {case.get('date_filed','')}", f"Date closed: {case.get('date_closed','')}", f"Patents: {', '.join(patent_vals)}", f"Parties: {', '.join(parties)}", f"Docket highlights: {'; '.join(docket[:10])}"]))
        md={"case": case, "documents": ds, "parties": ns, "patents": ps, "timeline": _timeline(case, ds), "source_files": source_files}
        ident=case.get("case_row_id") or case.get("case_number") or "unknown"
        docs.append(NormalizedDocument(f"litigation:{ident}", "litigation", case.get("case_name") or case.get("case_number") or ident, content, md))
    report["documents_parsed"]=len(docs); return docs, report
