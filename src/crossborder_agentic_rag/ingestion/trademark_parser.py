"""Trademark XML parser."""
from __future__ import annotations

import hashlib
from pathlib import Path
from xml.etree import ElementTree as ET

from crossborder_agentic_rag.ingestion.io_utils import ensure_input_dir, init_report
from crossborder_agentic_rag.schemas.documents import NormalizedDocument


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _norm(name: str) -> str:
    return name.lower().replace("_", "-")


def _text(e: ET.Element) -> str:
    return " ".join("".join(e.itertext()).split())


def find_first_text(element: ET.Element, candidate_names: list[str]) -> str:
    wanted = {_norm(n) for n in candidate_names}
    for child in element.iter():
        if _norm(local_name(child.tag)) in wanted:
            text = _text(child)
            if text:
                return text
    return ""


def find_all_text(element: ET.Element, candidate_names: list[str]) -> list[str]:
    wanted = {_norm(n) for n in candidate_names}
    values: list[str] = []
    for child in element.iter():
        if _norm(local_name(child.tag)) in wanted:
            text = _text(child)
            if text and text not in values:
                values.append(text)
    return values


def _records(root: ET.Element) -> list[ET.Element]:
    recs = [e for e in root.iter() if _norm(local_name(e.tag)) in {"case-file", "trademark-case-file", "casefile"}]
    return recs or [root]


def _goods_services(record: ET.Element) -> list[str]:
    values: list[str] = []
    for stmt in (e for e in record.iter() if _norm(local_name(e.tag)) == "case-file-statement"):
        type_code = find_first_text(stmt, ["type-code", "type_code"])
        if type_code.upper().startswith("GS"):
            for text in find_all_text(stmt, ["text", "statement-text", "statement_text"]):
                if text not in values:
                    values.append(text)
    for text in find_all_text(record, ["goods-services", "goods_services", "identification-of-goods", "identification_of_goods"]):
        if text not in values:
            values.append(text)
    return values


def _doc(record: ET.Element, path: Path, ordinal: int) -> NormalizedDocument | None:
    serial = find_first_text(record, ["serial-number", "serial_number"])
    registration = find_first_text(record, ["registration-number", "registration_number"])
    word = find_first_text(record, ["mark-identification", "word-mark", "word_mark"])
    filing = find_first_text(record, ["filing-date", "filing_date"])
    reg_date = find_first_text(record, ["registration-date", "registration_date"])
    status = find_first_text(record, ["status-code", "status_code"])
    goods = _goods_services(record)
    classes = find_all_text(record, ["international-code", "nice-class", "nice_classes", "nice-classification"])
    design = find_all_text(record, ["design-search-code", "design_search_code"])
    pseudo = find_all_text(record, ["pseudo-mark", "pseudo_mark"])
    fallback = hashlib.sha1(f"{path}:{ordinal}:{_text(record)[:200]}".encode()).hexdigest()[:12]
    ident = serial or registration or fallback
    metadata = {
        "source_subtype": "trademark_case_file", "serial_number": serial, "registration_number": registration, "word_mark": word,
        "mark_identification": word, "filing_date": filing, "registration_date": reg_date, "goods_services": goods,
        "nice_classes": classes, "status_code": status, "design_search_codes": design,
        "pseudo_marks": pseudo, "source_file": path.name, "source_path": str(path),
    }
    parts = [f"Word mark: {word}" if word else "", f"Serial number: {serial}" if serial else "", f"Registration number: {registration}" if registration else "", f"Goods/services: {'; '.join(goods)}" if goods else "", f"Nice classes: {', '.join(classes)}" if classes else "", f"Status code: {status}" if status else "", f"Design search codes: {', '.join(design)}" if design else "", f"Pseudo marks: {', '.join(pseudo)}" if pseudo else ""]
    content = "\n".join(p for p in parts if p)
    if len(content.strip()) < 10:
        return None
    return NormalizedDocument(f"trademark:{ident}", "trademark", word or serial or registration or ident, content, metadata)


def parse_trademark_xml_directory(input_dir: str | Path) -> tuple[list[NormalizedDocument], dict]:
    base = ensure_input_dir(input_dir)
    report = init_report("trademark", base)
    docs: list[NormalizedDocument] = []
    seen: set[str] = set()
    for path in sorted(base.rglob("*.xml")):
        report["files_seen"] += 1
        try:
            root = ET.parse(path).getroot()
            before = len(docs)
            for i, record in enumerate(_records(root)):
                doc = _doc(record, path, i)
                if doc is None:
                    report["warnings"].append({"path": str(path), "warning": "skipped empty trademark record"})
                    continue
                if doc.doc_id in seen:
                    report["warnings"].append({"path": str(path), "doc_id": doc.doc_id, "warning": "duplicate doc_id skipped"})
                    continue
                seen.add(doc.doc_id); docs.append(doc)
            report["files_parsed"] += 1
            if len(docs) == before:
                report["warnings"].append({"path": str(path), "warning": "no trademark records parsed"})
        except Exception as exc:
            report["failed_files"].append({"path": str(path), "error": str(exc)})
    report["documents_parsed"] = len(docs)
    return docs, report
