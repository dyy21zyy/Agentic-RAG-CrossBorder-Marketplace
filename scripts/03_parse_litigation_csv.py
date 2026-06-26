from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))
from crossborder_agentic_rag.ingestion.io_utils import write_documents_jsonl, write_report
from crossborder_agentic_rag.ingestion.litigation_parser import parse_litigation_csv_directory

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse litigation CSV files.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    try:
        docs, report = parse_litigation_csv_directory(Path(args.input))
        write_documents_jsonl(docs, args.output)
        write_report(report, args.report)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Parsed {report['documents_parsed']} documents from {report['files_parsed']}/{report['files_seen']} files.")
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
