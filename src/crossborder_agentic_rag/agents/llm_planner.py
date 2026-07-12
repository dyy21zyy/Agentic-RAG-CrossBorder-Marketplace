"""LLM planner for true Agentic RAG.

The planner decides:
1. which evidence tools should be called;
2. what retrieval query should be used for each tool;
3. which retrieval mode should be used for each tool call.

This makes the LLM-driven Agentic RAG different from a fixed pipeline:
- trademark risk can call trademark_search_tool with hybrid_rrf / hybrid_rerank;
- patent claim risk can call patent_search_tool with dense_only / hybrid_rerank;
- litigation exact lookup can call duckdb_lookup_tool + litigation_search_tool;
- complex entity relation can call graph_rag_tool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

from crossborder_agentic_rag.agents.retrieval_policy import (
    VALID_RETRIEVAL_MODES,
    DEFAULT_RETRIEVAL_MODE,
    normalize_retrieval_mode,
    build_rule_tool_plan,
)


CANONICAL_TOOL_NAMES = [
    "trademark_search_tool",
    "patent_search_tool",
    "litigation_search_tool",
    "duckdb_lookup_tool",
    "graph_rag_tool",
]

SOURCE_TO_TOOL = {
    "trademark": "trademark_search_tool",
    "patent": "patent_search_tool",
    "litigation": "litigation_search_tool",
    "structured": "duckdb_lookup_tool",
    "graph": "graph_rag_tool",
}

TOOL_TO_SOURCE = {
    "trademark_search_tool": "trademark",
    "patent_search_tool": "patent",
    "litigation_search_tool": "litigation",
    "duckdb_lookup_tool": "structured",
    "graph_rag_tool": "graph",
}


@dataclass(slots=True)
class ToolAction:
    tool: str
    query: str
    retrieval_mode: str | None = None
    reason: str = ""
    required_evidence: str | None = None


@dataclass(slots=True)
class LLMQueryPlan:
    query: str
    query_type: str
    expected_answer_type: str
    required_evidence: list[str] = field(default_factory=list)
    tool_plan: list[ToolAction] = field(default_factory=list)
    reason: str = ""
    planner_used: str = "heuristic"


def _safe_json_loads(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()

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

    raise ValueError("Cannot parse planner JSON")


def _call_llm_text(llm: Any, prompt: str) -> str:
    if hasattr(llm, "invoke"):
        resp = llm.invoke(prompt)
    elif hasattr(llm, "complete"):
        resp = llm.complete(prompt)
    elif hasattr(llm, "generate"):
        resp = llm.generate(prompt)
    elif callable(llm):
        resp = llm(prompt)
    else:
        raise TypeError("Unsupported LLM object")

    if isinstance(resp, str):
        return resp

    content = getattr(resp, "content", None)
    if content is not None:
        return str(content)

    if isinstance(resp, dict):
        for k in ["content", "text", "output", "answer"]:
            if k in resp:
                return str(resp[k])

    return str(resp)


def _normalize_tool_name(tool: str | None) -> str | None:
    if not tool:
        return None
    t = tool.strip()
    if t in CANONICAL_TOOL_NAMES:
        return t

    lower = t.lower()
    for name in CANONICAL_TOOL_NAMES:
        if lower == name.lower():
            return name

    aliases = {
        "trademark": "trademark_search_tool",
        "trademark_search": "trademark_search_tool",
        "patent": "patent_search_tool",
        "patent_search": "patent_search_tool",
        "litigation": "litigation_search_tool",
        "litigation_search": "litigation_search_tool",
        "duckdb": "duckdb_lookup_tool",
        "sql": "duckdb_lookup_tool",
        "structured": "duckdb_lookup_tool",
        "graph": "graph_rag_tool",
        "graphrag": "graph_rag_tool",
    }
    return aliases.get(lower)


def _normalize_evidence_type(value: str | None, tool: str | None = None) -> str | None:
    if value:
        v = value.strip().lower()
        if v in {"trademark", "patent", "litigation", "structured", "graph"}:
            return v
    if tool:
        return TOOL_TO_SOURCE.get(tool)
    return None


def heuristic_plan(query: str) -> LLMQueryPlan:
    """Fallback planner using deterministic rule-based adaptive plan.

    This is used when no LLM is provided or when LLM JSON parsing fails.
    """
    rule_plan = build_rule_tool_plan(query)

    actions = [
        ToolAction(
            tool=a.tool,
            query=a.query,
            retrieval_mode=a.retrieval_mode,
            reason=a.reason,
            required_evidence=a.source_type,
        )
        for a in rule_plan.actions
    ]

    required = []
    for a in actions:
        if a.required_evidence and a.required_evidence not in required:
            required.append(a.required_evidence)

    return LLMQueryPlan(
        query=query,
        query_type=rule_plan.query_type,
        expected_answer_type=rule_plan.expected_answer_type,
        required_evidence=required,
        tool_plan=actions,
        reason=rule_plan.reason,
        planner_used="heuristic_rule_tool_plan",
    )


def build_planner_prompt(query: str, available_tools: list[str] | None = None) -> str:
    tools = available_tools or CANONICAL_TOOL_NAMES
    tool_list = "\n".join(f"- {t}" for t in tools)

    retrieval_modes = ", ".join(sorted(VALID_RETRIEVAL_MODES))

    return f"""You are the planner of a true Agentic RAG system for cross-border e-commerce IP risk.

Your job is to decide:
1. which evidence tools should be called;
2. what retrieval query should be used for each tool;
3. which retrieval mode should be used for each tool call.

Available tools:
{tool_list}

Available retrieval modes:
{retrieval_modes}

Tool meanings:
- trademark_search_tool: trademark, word mark, brand, logo, Nice class, goods/services, counterfeit, confusing similarity.
- patent_search_tool: patent claims, patent descriptions, technical features, product-function matching, design/utility patent risk.
- litigation_search_tool: litigation case summaries, docket/case evidence, plaintiffs/defendants, asserted patent lawsuits.
- duckdb_lookup_tool: exact structured lookup, case numbers, docket IDs, patent numbers, trademark registration/serial numbers, exact word marks.
- graph_rag_tool: complex entity relationships, multi-hop reasoning, company-case-patent-brand connections.

Retrieval mode selection rules:
- bm25_only:
  Use for exact identifiers, case numbers, docket IDs, registration numbers, patent numbers, exact word marks, and keyword-heavy lookup.
- dense_only:
  Use for semantic patent claim matching, technical product features, paraphrased product descriptions, and broad semantic matching.
- hybrid_rrf:
  Use when both lexical matching and semantic recall are useful, especially trademark risk, multi-source recall, and entity-related search.
- hybrid_rerank:
  Use for high-precision final evidence, mixed IP risk, semantic litigation queries, trademark/logo similarity risk, and final decision support.

Important:
- Do not answer the user.
- Only output valid JSON.
- You may call multiple tools if the query needs multiple evidence types.
- For exact litigation/case/docket queries, prefer duckdb_lookup_tool first and litigation_search_tool second.
- For trademark risk, usually use trademark_search_tool with hybrid_rrf or hybrid_rerank.
- For patent claim risk, usually use patent_search_tool with dense_only or hybrid_rerank.
- For litigation semantic queries, usually use litigation_search_tool with hybrid_rerank.
- For complex entity relationship questions, include graph_rag_tool.
- For broad "Can I sell..." IP risk questions, include trademark, patent, and litigation tools if relevant.

Return JSON in this exact schema:
{{
  "query_type": "trademark | patent | litigation | mixed_ip_risk | entity_relation | exact_lookup | general_ip_question",
  "expected_answer_type": "trademark_answer | patent_answer | litigation_answer | risk_answer | graph_answer | structured_answer",
  "required_evidence": ["trademark", "patent", "litigation", "structured", "graph"],
  "tool_plan": [
    {{
      "tool": "one available tool name",
      "query": "retrieval-friendly query for this tool",
      "retrieval_mode": "bm25_only | dense_only | hybrid_rrf | hybrid_rerank",
      "reason": "why this tool and retrieval mode are needed",
      "required_evidence": "trademark | patent | litigation | structured | graph"
    }}
  ],
  "reason": "brief overall planning rationale"
}}

User query:
{query}
"""


def _plan_from_json(query: str, data: dict[str, Any]) -> LLMQueryPlan:
    query_type = str(data.get("query_type") or "general_ip_question")
    expected_answer_type = str(data.get("expected_answer_type") or "risk_answer")

    required_raw = data.get("required_evidence") or []
    required_evidence: list[str] = []
    if isinstance(required_raw, str):
        required_raw = [required_raw]
    if isinstance(required_raw, list):
        for x in required_raw:
            ev = _normalize_evidence_type(str(x))
            if ev and ev not in required_evidence:
                required_evidence.append(ev)

    tool_plan: list[ToolAction] = []
    raw_actions = data.get("tool_plan") or data.get("actions") or []

    if isinstance(raw_actions, dict):
        raw_actions = [raw_actions]

    if isinstance(raw_actions, list):
        for item in raw_actions:
            if not isinstance(item, dict):
                continue

            tool = _normalize_tool_name(item.get("tool"))
            if not tool:
                continue

            action_query = str(item.get("query") or query)
            evidence = _normalize_evidence_type(item.get("required_evidence"), tool=tool)

            retrieval_mode = item.get("retrieval_mode")
            if tool in {"duckdb_lookup_tool", "graph_rag_tool"}:
                # DuckDB and GraphRAG are not normal vector/BM25 retrieval backends.
                # Keep mode for GraphRAG only as metadata; DuckDB gets None.
                if tool == "duckdb_lookup_tool":
                    retrieval_mode = None
                else:
                    retrieval_mode = normalize_retrieval_mode(retrieval_mode, fallback="hybrid_rrf")
            else:
                retrieval_mode = normalize_retrieval_mode(retrieval_mode)

            tool_plan.append(
                ToolAction(
                    tool=tool,
                    query=action_query,
                    retrieval_mode=retrieval_mode,
                    reason=str(item.get("reason") or ""),
                    required_evidence=evidence,
                )
            )

            if evidence and evidence not in required_evidence:
                required_evidence.append(evidence)

    if not tool_plan:
        return heuristic_plan(query)

    return LLMQueryPlan(
        query=query,
        query_type=query_type,
        expected_answer_type=expected_answer_type,
        required_evidence=required_evidence,
        tool_plan=tool_plan,
        reason=str(data.get("reason") or ""),
        planner_used="llm",
    )


def plan_with_llm(
    query: str,
    llm: Any | None = None,
    available_tools: list[str] | None = None,
    tools: list[Any] | None = None,
    **kwargs: Any,
) -> LLMQueryPlan:
    """Plan tool calls and retrieval modes.

    Backward-compatible call styles:
    - plan_with_llm(query, llm=llm)
    - plan_with_llm(query, llm=llm, available_tools=[...])
    - plan_with_llm(query, llm=llm, tools=tools)

    If llm is None or LLM planning fails, fall back to deterministic rule-based adaptive planning.
    """
    if available_tools is None:
        available_tools = kwargs.get("available_tool_names")

    if available_tools is None and tools is not None:
        extracted = []
        for t in tools:
            name = getattr(t, "name", None) or getattr(t, "__name__", None)
            if name:
                extracted.append(str(name))
        available_tools = extracted or None

    if llm is None:
        return heuristic_plan(query)

    prompt = build_planner_prompt(query, available_tools=available_tools)

    try:
        raw = _call_llm_text(llm, prompt)
        data = _safe_json_loads(raw)
        return _plan_from_json(query, data)
    except Exception as exc:
        plan = heuristic_plan(query)
        plan.reason = f"{plan.reason} Fallback because LLM planner failed: {exc}"
        plan.planner_used = "heuristic_after_llm_failure"
        return plan

