"""Logical chunking utilities for normalized source documents."""

from __future__ import annotations

import re
from typing import Any

from crossborder_agentic_rag.schemas.documents import NormalizedDocument
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _as_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    return [value]


def _first(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def stable_chunk_id(doc_id: str, source_subtype: str, index: int | str) -> str:
    safe_doc_id = re.sub(r"[^A-Za-z0-9_.:-]+", "-", doc_id).strip("-")
    safe_subtype = re.sub(r"[^A-Za-z0-9_.:-]+", "-", source_subtype).strip("-")
    safe_index = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(index)).strip("-")
    return f"{safe_doc_id}:{safe_subtype}:{safe_index}"


def make_chunk(
    doc: NormalizedDocument,
    source_subtype: str,
    title: str,
    content: str,
    metadata_extra: dict | None = None,
    index: int | str = 0,
) -> EvidenceChunk:
    metadata = dict(doc.metadata)
    if metadata_extra:
        metadata.update(metadata_extra)
    return EvidenceChunk(
        chunk_id=stable_chunk_id(doc.doc_id, source_subtype, index),
        doc_id=doc.doc_id,
        source_type=doc.source_type,
        source_subtype=source_subtype,
        title=title,
        content=normalize_whitespace(content),
        metadata=metadata,
        score=0.0,
    )


def _append(chunks: list[EvidenceChunk], chunk: EvidenceChunk) -> None:
    if chunk.content.strip():
        chunks.append(chunk)


def chunk_document(doc: NormalizedDocument) -> list[EvidenceChunk]:
    if doc.source_type == "trademark":
        return chunk_trademark(doc)
    if doc.source_type == "patent":
        return chunk_patent(doc)
    if doc.source_type == "policy":
        return chunk_policy(doc)
    if doc.source_type == "litigation":
        return chunk_litigation(doc)
    raise ValueError(f"Unknown source_type for chunking: {doc.source_type}")


def chunk_documents(docs: list[NormalizedDocument]) -> list[EvidenceChunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc)]


def chunk_trademark(doc: NormalizedDocument) -> list[EvidenceChunk]:
    md = doc.metadata
    chunks: list[EvidenceChunk] = []
    serial = md.get("serial_number")
    reg = md.get("registration_number")
    word = md.get("word_mark")
    identity = f"Word mark: {word}\nSerial number: {serial}\nRegistration number: {reg}\nFiling date: {md.get('filing_date')}\nRegistration date: {md.get('registration_date')}\nStatus code: {md.get('status_code')}"
    _append(chunks, make_chunk(doc, "trademark_identity", f"{word or doc.title} identity", identity, {k: md.get(k) for k in ["serial_number", "registration_number", "word_mark", "filing_date", "registration_date", "status_code", "source_file", "source_path"]}, "identity"))

    classes = _as_list(md.get("nice_classes") or md.get("nice_class") or md.get("international_classes"))
    for i, cls in enumerate(classes or []):
        _append(chunks, make_chunk(doc, "trademark_class", f"{word or doc.title} class {cls}", f"Word mark {word} has Nice / international class {cls}.", {"nice_class": cls, "nice_classes": classes, "serial_number": serial, "registration_number": reg, "word_mark": word}, i))

    goods = _as_list(md.get("goods_services") or md.get("goods_and_services"))
    if goods:
        content = "\n".join(f"Goods/services: {g}" for g in goods)
        _append(chunks, make_chunk(doc, "trademark_goods_services", f"{word or doc.title} goods/services", content, {"goods_services": goods, "nice_classes": classes, "serial_number": serial, "registration_number": reg, "word_mark": word}, "goods"))

    design_codes = _as_list(md.get("design_search_codes"))
    pseudo = _as_list(md.get("pseudo_marks"))
    drawing = _first(md, "drawing_code", "drawing_description", "design_description")
    if design_codes or pseudo or drawing:
        content = f"Design search codes: {', '.join(map(str, design_codes))}\nPseudo marks: {', '.join(map(str, pseudo))}\nDrawing/design: {drawing or ''}"
        _append(chunks, make_chunk(doc, "trademark_design", f"{word or doc.title} design", content, {"design_search_codes": design_codes, "pseudo_marks": pseudo, "serial_number": serial, "registration_number": reg, "word_mark": word}, "design"))

    _append(chunks, make_chunk(doc, "trademark_record", f"{word or doc.title} record", doc.content or identity, {"serial_number": serial, "registration_number": reg, "word_mark": word}, "record"))
    return chunks or [make_chunk(doc, "trademark_record", doc.title, doc.content or doc.title, index="fallback")]


_CLAIM_RE = re.compile(r"(?ms)(?:^|\n)\s*(?:Claim\s*)?(\d+)\s*[\.)]\s+(.*?)(?=(?:\n\s*(?:Claim\s*)?\d+\s*[\.)]\s+)|\Z)")


def _split_claims(text: str) -> list[tuple[str, str]]:
    return [(m.group(1), normalize_whitespace(m.group(2))) for m in _CLAIM_RE.finditer(text or "") if normalize_whitespace(m.group(2))]


def _split_long_text(text: str, limit: int = 4000) -> list[str]:
    text = str(text or "").strip()
    if len(text) <= limit:
        return [text] if text else []
    parts: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", text):
        if len(current) + len(para) + 2 > limit and current:
            parts.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para).strip()
    if current:
        parts.append(current.strip())
    return parts


def chunk_patent(doc: NormalizedDocument) -> list[EvidenceChunk]:
    md = doc.metadata
    chunks: list[EvidenceChunk] = []
    patent_id = md.get("patent_id") or md.get("patent_number") or doc.doc_id
    claims_text = _first(md, "claims", "claim_text")
    claims = _split_claims(str(claims_text or ""))
    if claims_text and not claims:
        claims = [("unknown", str(claims_text))]
    for claim_no, text in claims:
        _append(chunks, make_chunk(doc, "patent_claim", f"Patent {patent_id} claim {claim_no}", f"Patent {patent_id} Claim {claim_no}:\n{text}", {"patent_id": patent_id, "claim_number": claim_no, "source_file": md.get("source_file"), "source_path": md.get("source_path")}, f"claim-{claim_no}"))

    summary = _first(md, "brief_summary", "summary")
    if summary:
        _append(chunks, make_chunk(doc, "patent_specification_summary", f"Patent {patent_id} summary", f"Patent {patent_id} summary:\n{summary}", {"patent_id": patent_id}, "summary"))
    detail = _first(md, "detail_description", "description")
    for i, part in enumerate(_split_long_text(str(detail or ""))):
        _append(chunks, make_chunk(doc, "patent_specification_detail", f"Patent {patent_id} detail {i + 1}", f"Patent {patent_id} detailed description:\n{part}", {"patent_id": patent_id, "part_index": i}, f"detail-{i}"))
    drawing = _first(md, "drawing_description", "drawings")
    if drawing:
        _append(chunks, make_chunk(doc, "patent_drawing", f"Patent {patent_id} drawings", f"Patent {patent_id} drawing description:\n{drawing}", {"patent_id": patent_id}, "drawing"))
    return chunks or [make_chunk(doc, "patent_specification_detail", doc.title, doc.content or doc.title, {"patent_id": patent_id}, "fallback")]


_HEADING_RE = re.compile(r"(?m)^\s*(#{1,6}\s+.+|\d+\.\s+[A-Z][^\n]+)$")
_CLAUSE_RE = re.compile(r"(?m)^\s*(?:\d+\.|[-*]|\([a-zA-Z]\))\s+(.+)$")
_ENFORCEMENT_TERMS = ("remove", "removal", "suspend", "suspension", "disable", "terminate", "penalty", "enforcement", "complaint", "counterfeit", "intellectual property complaint", "trademark infringement", "patent infringement")
_EXAMPLE_TERMS = ("example", "for example", "scenario", "case", "illustration")


def _sections(text: str) -> list[tuple[str, str]]:
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return []
    sections = []
    for i, m in enumerate(matches):
        title = re.sub(r"^#+\s*", "", m.group(1)).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((title, text[start:end].strip()))
    return sections


def chunk_policy(doc: NormalizedDocument) -> list[EvidenceChunk]:
    md = doc.metadata
    text = doc.content
    chunks: list[EvidenceChunk] = []
    platform = md.get("platform")
    policy_title = md.get("policy_title") or doc.title
    sections = _sections(text)
    if sections:
        for i, (title, body) in enumerate(sections):
            _append(chunks, make_chunk(doc, "policy_section", title, body, {"platform": platform, "policy_title": policy_title, "section_title": title, "source_file": md.get("source_file"), "source_path": md.get("source_path")}, i))
    else:
        _append(chunks, make_chunk(doc, "policy_section", doc.title, text, {"platform": platform, "policy_title": policy_title, "section_title": doc.title, "source_file": md.get("source_file"), "source_path": md.get("source_path")}, "fallback"))

    for i, m in enumerate(_CLAUSE_RE.finditer(text)):
        clause = m.group(0).strip()
        _append(chunks, make_chunk(doc, "policy_clause", f"{policy_title} clause {i + 1}", clause, {"platform": platform, "policy_title": policy_title}, i))
    for i, line in enumerate([ln for ln in text.splitlines() if any(term in ln.lower() for term in _ENFORCEMENT_TERMS)]):
        _append(chunks, make_chunk(doc, "policy_enforcement", f"{policy_title} enforcement {i + 1}", line, {"platform": platform, "policy_title": policy_title}, i))
    for i, line in enumerate([ln for ln in text.splitlines() if any(term in ln.lower() for term in _EXAMPLE_TERMS)]):
        _append(chunks, make_chunk(doc, "policy_example", f"{policy_title} example {i + 1}", line, {"platform": platform, "policy_title": policy_title}, i))
    return chunks


def chunk_litigation(doc: NormalizedDocument) -> list[EvidenceChunk]:
    md = doc.metadata
    case = md.get("case") if isinstance(md.get("case"), dict) else md
    chunks: list[EvidenceChunk] = []
    base = {"case_row_id": case.get("case_row_id"), "case_number": case.get("case_number"), "district_id": case.get("district_id")}
    summary = f"Case number: {case.get('case_number')}\nCase name: {case.get('case_name')}\nCourt name: {case.get('court_name')}\nDistrict id: {case.get('district_id')}\nDate filed: {case.get('date_filed')}\nDate closed: {case.get('date_closed')}\nCase cause: {case.get('case_cause')}"
    _append(chunks, make_chunk(doc, "litigation_case_summary", f"{case.get('case_name') or doc.title} summary", summary, {**base, "court_name": case.get("court_name"), "case_name": case.get("case_name"), "date_filed": case.get("date_filed"), "date_closed": case.get("date_closed"), "source_files": md.get("source_files")}, "summary"))
    for i, party in enumerate(_as_list(md.get("parties"))):
        if isinstance(party, dict):
            content = f"Party type: {party.get('party_type')}\nName: {party.get('name')}\nName long: {party.get('name_long')}\nCase number: {case.get('case_number')}\nCase name: {case.get('case_name')}"
            _append(chunks, make_chunk(doc, "litigation_party", f"{case.get('case_name')} party {party.get('name')}", content, {**base, "party_type": party.get("party_type"), "name": party.get("name"), "name_long": party.get("name_long")}, i))
    for i, patent in enumerate(_as_list(md.get("patents"))):
        pnum = patent.get("patent_number") or patent.get("patent") if isinstance(patent, dict) else patent
        content = f"Patent number: {pnum}\nCase number: {case.get('case_number')}\nCase name: {case.get('case_name')}\nCase type: {case.get('case_type')}\nDate filed: {case.get('date_filed')}"
        _append(chunks, make_chunk(doc, "litigation_patent", f"{case.get('case_name')} patent {pnum}", content, {**base, "patent": patent, "patent_number": pnum}, i))
    for i, docket in enumerate(_as_list(md.get("documents"))):
        if isinstance(docket, dict):
            date_filed = docket.get("doc_date_filed") or docket.get("date_filed")
            content = f"Doc number: {docket.get('doc_number')}\nShort description: {docket.get('short_description')}\nLong description: {docket.get('long_description')}\nDocument filed date: {date_filed}\nCase number: {case.get('case_number')}"
            _append(chunks, make_chunk(doc, "litigation_docket", f"{case.get('case_name')} docket {docket.get('doc_number')}", content, {**base, "doc_number": docket.get("doc_number"), "doc_date_filed": date_filed, "short_description": docket.get("short_description")}, i))
    timeline = _as_list(md.get("timeline"))
    if timeline:
        rows = []
        for event in sorted(timeline, key=lambda e: e.get("date", "") if isinstance(e, dict) else str(e)):
            rows.append(f"{event.get('date')}: {event.get('event_type')} - {event.get('description')}" if isinstance(event, dict) else str(event))
        _append(chunks, make_chunk(doc, "litigation_timeline", f"{case.get('case_name')} timeline", "\n".join(rows), {**base, "timeline": timeline}, "timeline"))
    return chunks or [make_chunk(doc, "litigation_case_summary", doc.title, doc.content or doc.title, base, "fallback")]
