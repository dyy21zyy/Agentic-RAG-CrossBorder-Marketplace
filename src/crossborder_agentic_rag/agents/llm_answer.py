"""Grounded LLM answer construction shared by runtime and evaluation."""
from __future__ import annotations
from typing import Any


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def build_evidence_context(chunks, max_evidence: int = 6, max_chars_each: int = 450) -> tuple[str, list[dict]]:
    lines: list[str] = []
    manifest: list[dict] = []
    for i, c in enumerate(list(chunks or [])[:max_evidence], start=1):
        content = str(_get(c, "content", _get(c, "content_preview", "")) or "")[:max_chars_each]
        item = {
            "evidence_id": f"E{i}",
            "chunk_id": _get(c, "chunk_id"),
            "doc_id": _get(c, "doc_id"),
            "source_type": _get(c, "source_type"),
            "source_subtype": _get(c, "source_subtype"),
            "title": _get(c, "title"),
            "score": _get(c, "score"),
            "content": content,
        }
        manifest.append(item)
        lines.append(f"[E{i}] chunk_id={item['chunk_id']} doc_id={item['doc_id']} source_type={item['source_type']} source_subtype={item['source_subtype']} title={item['title']} score={item['score']}\n{content}")
    return "\n\n".join(lines), manifest


def build_grounded_answer_messages(query: str, evidence_context: str, query_type: str | None = None, retrieval_route: str | None = None, evidence_gaps: list[str] | None = None, pipeline_mode: str | None = None, source_scope: list[str] | None = None) -> list[dict[str, str]]:
    system = (
        "You are a retrieval-grounded compliance research assistant. This is not legal advice. "
        "Answer only from the provided evidence. Do not invent trademarks, patents, registrations, lawsuits, claims, infringement facts, legal conclusions, or source details. "
        "Cite evidence using [E1], [E2], etc. If evidence is insufficient, explicitly say what is missing. "
        "The current project focuses on trademark, patent, and litigation evidence. Do not require policy evidence unless the user explicitly asks about platform policy. "
        "Keep answer clear and useful for marketplace seller compliance research."
    )
    meta = []
    if pipeline_mode: meta.append(f"pipeline_mode: {pipeline_mode}")
    if query_type: meta.append(f"query_type: {query_type}")
    if retrieval_route: meta.append(f"retrieval_route: {retrieval_route}")
    if source_scope: meta.append(f"source_scope: {', '.join(source_scope)}")
    if evidence_gaps: meta.append("evidence_gaps: " + "; ".join(evidence_gaps))
    user = f"User query: {query}\n" + ("\n".join(meta) + "\n" if meta else "") + f"\nEvidence:\n{evidence_context or '(none)'}\n\nRequired answer format:\n1. Direct answer\n2. Key evidence\n3. Risk analysis if applicable\n4. Suggested seller actions\n5. Evidence citations\n6. Limitations / missing evidence"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_grounded_answer(query: str, evidence_chunks, chat_client, query_type: str | None = None, retrieval_route: str | None = None, evidence_gaps: list[str] | None = None, pipeline_mode: str | None = None, max_evidence: int = 6, max_chars_each: int = 450, max_tokens: int = 800, temperature: float = 0.0) -> dict:
    evidence_context, manifest = build_evidence_context(evidence_chunks, max_evidence, max_chars_each)
    result = {"llm_answer": "", "llm_error": None, "evidence_manifest": manifest, "max_evidence": max_evidence, "max_chars_each": max_chars_each, "provider": None, "model": None}
    if not manifest:
        result["llm_error"] = "No evidence available for grounded answer"
        return result
    try:
        messages = build_grounded_answer_messages(query, evidence_context, query_type, retrieval_route, evidence_gaps, pipeline_mode)
        chat = chat_client.complete(messages, temperature=temperature, max_tokens=max_tokens)
        result.update({"provider": chat.provider, "model": chat.model})
        if chat.error:
            result["llm_error"] = chat.error
        else:
            result["llm_answer"] = chat.content or ""
            if not chat.content:
                result["llm_error"] = "LLM response content is empty"
    except Exception as exc:
        result["llm_error"] = str(exc)
    return result
