"""Patent litigation CSV parser."""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from crossborder_agentic_rag.ingestion.io_utils import ensure_input_dir, init_report
from crossborder_agentic_rag.schemas.documents import NormalizedDocument

TABLES = ("cases", "documents", "names", "patents")
CASE_IDS=["case_id","case_row_id","case_number","docket_number","civil_action_number"]
COURTS=["court","court_name","district_court","district_id"]
DATES=["filing_date","date_filed","filed_date"]
PATENTS=["patent_id","asserted_patent","patent","patent_number"]
PARTIES=["plaintiff","defendant","party_name","assignee","owner","name_long","name"]

def _key(name:str)->str: return str(name).strip().lower().replace(" ","_").replace("-","_")
def _get(row:dict,names:list[str])->str:
    low={_key(k):v for k,v in row.items()}
    for n in names:
        v=row.get(n) or low.get(_key(n))
        if v and str(v).strip(): return str(v).strip()
    return ""

def litigation_case_key(row: dict) -> tuple[str, str, str]:
    return (_get(row, CASE_IDS), _get(row, ["case_number","docket_number","civil_action_number"]), _get(row, ["district_id", *COURTS]))

def _kind(path: Path) -> str | None:
    n = path.name.lower()
    return next((t for t in TABLES if t in n), None)

def _read(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]

def _timeline(case, docs):
    events=[]; filed=_get(case,DATES); closed=_get(case,["date_closed","closed_date"])
    if filed: events.append({"date": filed, "event_type": "filed", "description": "Case filed"})
    if closed: events.append({"date": closed, "event_type": "closed", "description": "Case closed"})
    for d in docs:
        date=_get(d,["doc_date_filed","date_filed","filing_date"])
        if date: events.append({"date": date, "event_type": "docket", "description": _get(d,["short_description","description"])})
    return sorted(events, key=lambda e: e.get("date") or "9999")

def parse_litigation_csv_directory(input_dir: str | Path) -> tuple[list[NormalizedDocument], dict]:
    base=ensure_input_dir(input_dir); report=init_report("litigation", base); rows={t: [] for t in TABLES}; files={t: [] for t in TABLES}; docs=[]
    for path in sorted(base.rglob("*.csv")):
        kind=_kind(path)
        if not kind: continue
        report["files_seen"] += 1
        try: rows[kind].extend(_read(path)); files[kind].append(path.name); report["files_parsed"] += 1
        except Exception as exc: report["failed_files"].append({"path": str(path), "error": str(exc)})
    if not rows["cases"]: report["warnings"].append({"warning": "no cases CSV rows found"})
    for t in ("documents", "names", "patents"):
        if not rows[t]: report["warnings"].append({"warning": f"missing related table: {t}"})
    grouped={t: defaultdict(list) for t in ("documents", "names", "patents")}
    for t in grouped:
        for r in rows[t]: grouped[t][litigation_case_key(r)].append(r)
    source_files=sorted({f for t in TABLES for f in files[t]})
    seen=set()
    for case in rows["cases"]:
        key=litigation_case_key(case); ds=grouped["documents"].get(key, []); ps=grouped["patents"].get(key, []); ns=grouped["names"].get(key, [])
        case_id=_get(case, CASE_IDS); case_number=_get(case,["case_number","docket_number","civil_action_number"]); court=_get(case, COURTS); filing=_get(case,DATES)
        patent_vals=[_get(p,PATENTS) for p in ps if _get(p,PATENTS)]
        parties=[_get(n,PARTIES) for n in ns if _get(n,PARTIES)]
        plaintiff=_get(case,["plaintiff"]); defendant=_get(case,["defendant"])
        if plaintiff: parties.append(plaintiff)
        if defendant: parties.append(defendant)
        parties=list(dict.fromkeys(parties)); docket=[_get(d,["short_description","description"]) for d in ds if _get(d,["short_description","description"])]
        content="\n".join(filter(None,[f"Case number: {case_number or case_id}", f"Case name: {_get(case,['case_name','name'])}", f"Court: {court}", f"Filing date: {filing}", f"Plaintiff: {plaintiff}", f"Defendant: {defendant}", f"Asserted patents: {', '.join(patent_vals)}" if patent_vals else "", f"Parties: {', '.join(parties)}" if parties else "", f"Description: {_get(case,['description','short_description'])}", f"Docket highlights: {'; '.join(docket[:10])}" if docket else ""]))
        md={"source_subtype":"patent_litigation_case","case_id":case_id,"case_number":case_number,"court":court,"filing_date":filing,"plaintiff":plaintiff,"defendant":defendant,"patent_numbers":patent_vals,"case":case,"documents":ds,"parties":ns,"patents":ps,"timeline":_timeline(case,ds),"source_files":source_files,"source_file":source_files[0] if source_files else "","source_path":str(base)}
        ident=case_id or case_number or "unknown"
        doc_id=f"litigation:{ident}"
        if doc_id in seen: report["warnings"].append({"doc_id":doc_id,"warning":"duplicate doc_id skipped"}); continue
        seen.add(doc_id); docs.append(NormalizedDocument(doc_id, "litigation", _get(case,["case_name","name"]) or case_number or ident, content, md))
    report["documents_parsed"]=len(docs); return docs, report
