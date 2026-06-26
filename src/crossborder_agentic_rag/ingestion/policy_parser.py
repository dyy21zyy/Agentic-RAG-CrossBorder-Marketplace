"""Policy document parser."""
from __future__ import annotations
import hashlib
from html.parser import HTMLParser
from pathlib import Path
from crossborder_agentic_rag.ingestion.io_utils import ensure_input_dir, init_report
from crossborder_agentic_rag.schemas.documents import NormalizedDocument

SUPPORTED={".txt", ".md", ".html", ".htm", ".pdf"}
class _Stripper(HTMLParser):
    def __init__(self): super().__init__(); self.parts=[]
    def handle_data(self, data):
        if data.strip(): self.parts.append(data.strip())

def _strip_html(raw: str) -> str:
    s=_Stripper(); s.feed(raw); return " ".join(s.parts)

def _read_html(path: Path, report: dict) -> str:
    raw=path.read_text(encoding="utf-8", errors="replace")
    try:
        from bs4 import BeautifulSoup  # type: ignore
        return BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
    except ImportError:
        report["warnings"].append({"path": str(path), "warning": "BeautifulSoup not installed; used simple HTML stripping fallback"})
        return _strip_html(raw)

def _read_pdf(path: Path, report: dict) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        report["warnings"].append({"path": str(path), "warning": "pypdf not installed; skipped PDF"}); return ""
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)

def _title(path: Path, text: str) -> str:
    for line in text.splitlines():
        clean=line.strip().lstrip("# ").strip()
        if clean: return clean[:200]
    return path.stem

def parse_policy_directory(input_dir: str | Path) -> tuple[list[NormalizedDocument], dict]:
    base=ensure_input_dir(input_dir); report=init_report("policy", base); docs=[]
    for path in sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED):
        report["files_seen"] += 1
        try:
            suffix=path.suffix.lower()
            if suffix in {".txt", ".md"}: text=path.read_text(encoding="utf-8", errors="replace")
            elif suffix in {".html", ".htm"}: text=_read_html(path, report)
            else: text=_read_pdf(path, report)
            text="\n".join(line.strip() for line in text.splitlines() if line.strip())
            if not text.strip():
                report["warnings"].append({"path": str(path), "warning": "skipped empty policy file"}); continue
            title=_title(path, text); platform="Temu" if "temu" in str(path).lower() else "unknown"
            ident=f"{path.stem}-{hashlib.sha1(str(path.relative_to(base)).encode()).hexdigest()[:10]}"
            md={"platform": platform, "policy_title": title, "source_file": path.name, "source_path": str(path), "file_type": suffix}
            docs.append(NormalizedDocument(f"policy:{ident}", "policy", title, text, md)); report["files_parsed"] += 1
        except Exception as exc:
            report["failed_files"].append({"path": str(path), "error": str(exc)})
    report["documents_parsed"]=len(docs); return docs, report
