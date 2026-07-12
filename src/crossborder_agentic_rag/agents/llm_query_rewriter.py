"""LLM query rewriter for evidence-gap driven follow-up retrieval.

This module rewrites follow-up queries when the current evidence is insufficient.

It supports:
- LLM mode: use an LLM/chat client to rewrite retrieval queries.
- Heuristic fallback mode: use deterministic templates when LLM is unavailable.

The rewriter does not execute retrieval. It only produces follow-up queries.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any


EVIDENCE_TO_TOOL = {
    "trademark": "trademark_search_tool",
    "patent": "patent_search_tool",
    "litigation": "litigation_search_tool",
    "structured": "duckdb_lookup_tool",
    "graph": "graph_rag_tool",
}


@dataclass(slots=True)
class FollowupQuery:
    """One rewritten follow-up retrieval query."""

    missing_evidence: str
    tool: str
    query: str
    reason: str = ""


@dataclass(slots=True)
class QueryRewritePlan:
    """A set of follow-up queries for missing evidence."""

    original_query: str
    followup_queries: list[FollowupQuery] = field(default_factory=list)
    rationale: str = ""
    source: str = "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_text(value: Any, max_chars: int = 800) -> str:
    if value is None:
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _chunk_to_summary(item: Any, max_chars: int = 260) -> str:
    """Convert evidence object or dict into compact text for prompting."""

    if isinstance(item, dict):
        chunk_id = item.get("chunk_id") or item.get("doc_id") or ""
        source_type = item.get("source_type") or ""
        title = item.get("title") or ""
        content = item.get("content") or item.get("content_preview") or ""
        score = item.get("score")
        return _safe_text(
            f"[{source_type}] {title} {chunk_id} score={score}: {content}",
            max_chars=max_chars,
        )

    chunk_id = getattr(item, "chunk_id", "") or getattr(item, "doc_id", "")
    source_type = getattr(item, "source_type", "")
    title = getattr(item, "title", "")
    content = getattr(item, "content", "")
    score = getattr(item, "score", None)

    return _safe_text(
        f"[{source_type}] {title} {chunk_id} score={score}: {content}",
        max_chars=max_chars,
    )


def summarize_evidence(evidence: list[Any] | None, max_items: int = 6, max_chars_each: int = 260) -> str:
    """Summarize current evidence for LLM query rewriting."""

    if not evidence:
        return "No evidence retrieved yet."

    lines: list[str] = []
    for idx, item in enumerate(evidence[:max_items], start=1):
        lines.append(f"{idx}. {_chunk_to_summary(item, max_chars=max_chars_each)}")

    return "\n".join(lines)


def heuristic_rewrite_query(original_query: str, missing_evidence: str) -> str:
    """Deterministic fallback query rewrite for one missing evidence type."""

    m = (missing_evidence or "").lower().strip()

    if m == "trademark":
        return (
            "trademark brand logo word mark Nice class goods services "
            f"evidence for: {original_query}"
        )

    if m == "patent":
        return (
            "patent claim product feature invention utility design patent "
            f"evidence for: {original_query}"
        )

    if m == "litigation":
        return (
            "litigation lawsuit case docket plaintiff defendant party asserted patent "
            f"evidence for: {original_query}"
        )

    if m == "structured":
        return (
            "exact lookup registration number serial number word mark patent number "
            f"case number structured metadata for: {original_query}"
        )

    if m == "graph":
        return (
            "entity relationship company trademark patent case party graph connection "
            f"evidence for: {original_query}"
        )

    return f"Find {missing_evidence} evidence for: {original_query}"


def heuristic_rewrite(
    original_query: str,
    evidence_gaps: list[str] | None = None,
    missing_evidence: list[str] | None = None,
    max_queries: int = 5,
) -> QueryRewritePlan:
    """Create follow-up queries without LLM."""

    missing = missing_evidence or evidence_gaps or []
    cleaned: list[str] = []

    for item in missing:
        text = str(item).lower().strip()
        text = text.replace("missing ", "").replace(" evidence", "").strip()
        if text and text not in cleaned:
            cleaned.append(text)

    followups: list[FollowupQuery] = []
    for m in cleaned[:max_queries]:
        tool = EVIDENCE_TO_TOOL.get(m, "trademark_search_tool")
        followups.append(
            FollowupQuery(
                missing_evidence=m,
                tool=tool,
                query=heuristic_rewrite_query(original_query, m),
                reason=f"Heuristic follow-up query for missing {m} evidence.",
            )
        )

    return QueryRewritePlan(
        original_query=original_query,
        followup_queries=followups,
        rationale="Heuristic query rewrite based on missing evidence types.",
        source="heuristic",
    )


def build_rewriter_prompt(
    original_query: str,
    evidence_gaps: list[str],
    current_evidence_summary: str,
    available_tool_names: list[str] | None = None,
) -> str:
    tools = available_tool_names or list(EVIDENCE_TO_TOOL.values())

    return f"""You are an IP retrieval query rewriting agent.

The current evidence is insufficient. Rewrite follow-up queries to retrieve the missing evidence.

Original user query:
{original_query}

Current retrieved evidence summary:
{current_evidence_summary}

Evidence gaps:
{evidence_gaps}

Available tools:
{tools}

Return JSON only. Do not include markdown.

JSON schema:
{{
  "followup_queries": [
    {{
      "missing_evidence": "trademark | patent | litigation | structured | graph",
      "tool": "one available tool name",
      "query": "retrieval-friendly follow-up query",
      "reason": "why this rewritten query should retrieve the missing evidence"
    }}
  ],
  "rationale": "brief reason for the rewrite plan"
}}

Rules:
- Generate only queries for missing evidence.
- Make each query retrieval-friendly, specific, and concise.
- Do not simply repeat the original query unless it is already specific.
- If trademark evidence is missing, include terms such as trademark, brand, logo, word mark, Nice class, goods/services.
- If patent evidence is missing, include terms such as patent claim, product feature, invention, utility patent, design patent.
- If litigation evidence is missing, include terms such as litigation, lawsuit, case, docket, party, plaintiff, defendant, asserted patent.
- If structured evidence is missing, preserve exact identifiers from the original query.
- If graph evidence is missing, focus on entity relationships such as company-to-case, case-to-patent, owner-to-trademark, or party-to-patent.
- Do not invent facts. Only rewrite search queries.
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text).strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return json.loads(cleaned[start : end + 1])

    raise ValueError("No valid JSON object found in query rewriter output.")


def _call_llm_text(llm: Any, prompt: str) -> str:
    """Call different possible LLM client interfaces and return text."""

    if llm is None:
        raise ValueError("llm is None")

    if hasattr(llm, "invoke"):
        response = llm.invoke(prompt)
    elif hasattr(llm, "complete"):
        response = llm.complete(prompt)
    elif hasattr(llm, "generate"):
        response = llm.generate(prompt)
    elif callable(llm):
        response = llm(prompt)
    else:
        raise TypeError("Unsupported LLM interface for query rewriter.")

    if isinstance(response, str):
        return response

    content = getattr(response, "content", None)
    if content is not None:
        return str(content)

    if isinstance(response, dict):
        for key in ["content", "text", "output", "answer"]:
            if key in response:
                return str(response[key])

    return str(response)


def _normalize_tool_name(tool: str, missing_evidence: str, available_tool_names: list[str]) -> str:
    if tool in available_tool_names:
        return tool

    fallback = EVIDENCE_TO_TOOL.get(missing_evidence, "trademark_search_tool")
    if fallback in available_tool_names:
        return fallback

    return available_tool_names[0] if available_tool_names else fallback


def _coerce_rewrite_plan(
    original_query: str,
    data: dict[str, Any],
    evidence_gaps: list[str],
    available_tool_names: list[str],
    max_queries: int,
) -> QueryRewritePlan:
    items = data.get("followup_queries") or []
    followups: list[FollowupQuery] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        missing = str(item.get("missing_evidence") or "").lower().strip()
        if not missing:
            continue

        raw_tool = str(item.get("tool") or "")
        tool = _normalize_tool_name(raw_tool, missing, available_tool_names)
        query = str(item.get("query") or heuristic_rewrite_query(original_query, missing)).strip()
        reason = str(item.get("reason") or "")

        followups.append(
            FollowupQuery(
                missing_evidence=missing,
                tool=tool,
                query=query,
                reason=reason,
            )
        )

    if not followups:
        return heuristic_rewrite(
            original_query=original_query,
            evidence_gaps=evidence_gaps,
            max_queries=max_queries,
        )

    return QueryRewritePlan(
        original_query=original_query,
        followup_queries=followups[:max_queries],
        rationale=str(data.get("rationale") or "LLM-generated follow-up retrieval queries."),
        source="llm",
    )


def rewrite_with_llm(
    original_query: str,
    evidence_gaps: list[str] | None = None,
    current_evidence: list[Any] | None = None,
    llm: Any | None = None,
    available_tool_names: list[str] | None = None,
    max_queries: int = 5,
) -> QueryRewritePlan:
    """Rewrite follow-up queries using LLM when available, otherwise fallback."""

    gaps = evidence_gaps or []
    tools = available_tool_names or list(EVIDENCE_TO_TOOL.values())

    if not gaps:
        return QueryRewritePlan(
            original_query=original_query,
            followup_queries=[],
            rationale="No evidence gaps were provided.",
            source="none",
        )

    if llm is None:
        return heuristic_rewrite(
            original_query=original_query,
            evidence_gaps=gaps,
            max_queries=max_queries,
        )

    current_summary = summarize_evidence(current_evidence)
    prompt = build_rewriter_prompt(
        original_query=original_query,
        evidence_gaps=gaps,
        current_evidence_summary=current_summary,
        available_tool_names=tools,
    )

    try:
        raw = _call_llm_text(llm, prompt)
        data = _extract_json_object(raw)
        return _coerce_rewrite_plan(
            original_query=original_query,
            data=data,
            evidence_gaps=gaps,
            available_tool_names=tools,
            max_queries=max_queries,
        )
    except Exception as exc:
        fallback = heuristic_rewrite(
            original_query=original_query,
            evidence_gaps=gaps,
            max_queries=max_queries,
        )
        fallback.rationale = f"LLM query rewriter failed; used heuristic fallback. Error: {exc}"
        fallback.source = "heuristic_after_llm_failure"
        return fallback
