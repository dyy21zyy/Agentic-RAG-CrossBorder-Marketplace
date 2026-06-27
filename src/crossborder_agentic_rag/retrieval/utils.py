"""Runtime retrieval formatting and de-duplication helpers."""
from __future__ import annotations
from collections import Counter
from typing import Any, Iterable


def _get(chunk: Any, name: str, default: Any = None) -> Any:
    if isinstance(chunk, dict):
        return chunk.get(name, default)
    return getattr(chunk, name, default)


def _fallback_key(chunk: Any) -> str:
    doc_id = str(_get(chunk, "doc_id", "") or "")
    title = str(_get(chunk, "title", "") or "")
    content = str(_get(chunk, "content", "") or "")[:200]
    return "fallback:" + "|".join([doc_id, title, content])


def dedupe_chunks(chunks: Iterable[Any], key: str = "chunk_id") -> list[Any]:
    """Return chunks with duplicates removed while preserving first occurrence order."""
    out: list[Any] = []
    seen: set[str] = set()
    for i, chunk in enumerate(chunks):
        value = _get(chunk, key, None)
        dedupe_key = f"{key}:{value}" if value not in (None, "") else _fallback_key(chunk)
        if dedupe_key == "fallback:||":
            dedupe_key = f"object:{i}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        out.append(chunk)
    return out


def summarize_source_counts(chunks: Iterable[Any]) -> dict[str, dict[str, int]]:
    """Summarize source_type/source_subtype counts for objects or dict chunks."""
    source_type_counts: Counter[str] = Counter()
    source_subtype_counts: Counter[str] = Counter()
    for chunk in chunks:
        source_type = _get(chunk, "source_type", None)
        source_subtype = _get(chunk, "source_subtype", None)
        if source_type:
            source_type_counts[str(source_type)] += 1
        if source_subtype:
            source_subtype_counts[str(source_subtype)] += 1
    return {
        "source_type_counts": dict(source_type_counts),
        "source_subtype_counts": dict(source_subtype_counts),
    }


def evidence_to_dict(chunk: Any, rank: int | None = None, preview_chars: int = 500) -> dict[str, Any]:
    """Convert an EvidenceChunk-like object into a compact JSON-serializable dict."""
    metadata = _get(chunk, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    content = str(_get(chunk, "content", "") or "")
    preview_len = max(0, int(preview_chars))
    score = _get(chunk, "score", None)
    try:
        score = None if score is None else float(score)
    except (TypeError, ValueError):
        score = None
    return {
        "rank": rank,
        "chunk_id": _get(chunk, "chunk_id", None),
        "doc_id": _get(chunk, "doc_id", None),
        "source_type": _get(chunk, "source_type", None),
        "source_subtype": _get(chunk, "source_subtype", None),
        "title": _get(chunk, "title", None),
        "score": score,
        "metadata": metadata,
        "content_preview": content[:preview_len],
    }
