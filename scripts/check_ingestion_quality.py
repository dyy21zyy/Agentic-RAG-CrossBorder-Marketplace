from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path


def _load_json(line: str, line_no: int, report: dict):
    try: return json.loads(line)
    except json.JSONDecodeError as exc:
        report["json_parse_failures"] += 1; report["warnings"].append({"line": line_no, "warning": str(exc)}); return None


def check_quality(input_path: str | Path, require_source_types: str = "trademark,patent,litigation", sample_size: int = 5) -> dict:
    path=Path(input_path)
    report={"input":str(path),"total_records":0,"counts_by_source_type":{},"counts_by_source_subtype":{},"empty_content_count":0,"very_short_content_count":0,"duplicate_doc_id_count":0,"duplicate_chunk_id_count":0,"missing_title_count":0,"missing_metadata_count":0,"metadata_json_parse_failures":0,"json_parse_failures":0,"sample_records_by_source_type":{},"missing_required_source_types":[],"warnings":[]}
    if not path.exists():
        report["fatal_error"]="input missing"; return report
    src=Counter(); sub=Counter(); samples=defaultdict(list); doc_ids=set(); chunk_ids=set()
    with path.open("r",encoding="utf-8",errors="replace") as fh:
        for i,line in enumerate(fh,1):
            if not line.strip(): continue
            rec=_load_json(line,i,report)
            if rec is None: continue
            report["total_records"]+=1
            st=rec.get("source_type") or rec.get("metadata",{}).get("source_type") or ""
            ss=rec.get("source_subtype") or rec.get("metadata",{}).get("source_subtype") or ""
            if st: src[st]+=1
            if ss: sub[ss]+=1
            content=str(rec.get("content") or "")
            if not content.strip(): report["empty_content_count"]+=1
            elif len(content.strip())<30: report["very_short_content_count"]+=1
            if not str(rec.get("title") or "").strip(): report["missing_title_count"]+=1
            if not isinstance(rec.get("metadata"), dict): report["missing_metadata_count"]+=1
            if rec.get("metadata_json"):
                try: json.loads(rec["metadata_json"])
                except Exception: report["metadata_json_parse_failures"]+=1
            did=rec.get("doc_id")
            if did:
                if did in doc_ids: report["duplicate_doc_id_count"]+=1
                doc_ids.add(did)
            cid=rec.get("chunk_id")
            if cid:
                if cid in chunk_ids: report["duplicate_chunk_id_count"]+=1
                chunk_ids.add(cid)
            if st and len(samples[st])<sample_size:
                samples[st].append({k:rec.get(k) for k in ("doc_id","chunk_id","source_type","source_subtype","title") if rec.get(k) is not None})
    required=[s.strip() for s in require_source_types.split(",") if s.strip()]
    report["counts_by_source_type"]=dict(src); report["counts_by_source_subtype"]=dict(sub); report["sample_records_by_source_type"]=dict(samples)
    report["missing_required_source_types"]=[s for s in required if src.get(s,0)==0]
    return report


def main(argv: list[str] | None = None) -> int:
    ap=argparse.ArgumentParser(description="Check normalized document or chunk JSONL ingestion quality.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--require-source-types", default="trademark,patent,litigation")
    ap.add_argument("--sample-size", type=int, default=5)
    ap.add_argument("--output-json")
    ap.add_argument("--allow-warnings", action="store_true")
    args=ap.parse_args(argv)
    report=check_quality(args.input,args.require_source_types,args.sample_size)
    text=json.dumps(report,indent=2,sort_keys=True)
    if args.output_json:
        out=Path(args.output_json); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(text+"\n",encoding="utf-8")
    print(text)
    fatal=bool(report.get("fatal_error")) or report["total_records"]==0
    warning_fail=bool(report["missing_required_source_types"] or report["duplicate_doc_id_count"] or report["duplicate_chunk_id_count"] or report["metadata_json_parse_failures"])
    if fatal: return 1
    if warning_fail and not args.allow_warnings: return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
