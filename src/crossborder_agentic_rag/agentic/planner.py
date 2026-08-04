from typing import Any

from crossborder_agentic_rag.llm.chat_client import BaseChatClient

from .rewriter import rewrite_for_scenario


TOOL_BY_SCOPE = {
    "trademark": "trademark_search_tool",
    "patent": "patent_search_tool",
    "litigation": "litigation_search_tool",
}


def _valid_llm_steps(structured: Any, scope: list[str]) -> list[dict[str, object]]:
    if not isinstance(structured, dict) or not isinstance(structured.get("tool_plan"), list):
        return []

    allowed = {TOOL_BY_SCOPE[item]: item for item in scope if item in TOOL_BY_SCOPE}
    valid: list[dict[str, object]] = []
    for raw_step in structured["tool_plan"]:
        if not isinstance(raw_step, dict):
            continue
        tool = raw_step.get("tool")
        query = raw_step.get("query")
        retrieval_mode = raw_step.get("retrieval_mode")
        required_evidence = raw_step.get("required_evidence")
        if not all(isinstance(value, str) for value in (tool, query, retrieval_mode, required_evidence)):
            continue
        tool = tool.strip()
        query = query.strip()
        retrieval_mode = retrieval_mode.strip()
        required_evidence = required_evidence.strip()
        if (
            not tool
            or not query
            or not required_evidence
            or allowed.get(tool) != required_evidence
            or retrieval_mode != "hybrid_rerank"
        ):
            continue
        valid.append(
            {
                "tool": tool,
                "query": query,
                "retrieval_mode": retrieval_mode,
                "required_evidence": required_evidence,
            }
        )
    return valid


def plan_tools(query: str, scope: list[str], llm: BaseChatClient | None = None) -> list[dict[str, object]]:
    if llm is not None:
        allowed_tools = [TOOL_BY_SCOPE[item] for item in scope if item in TOOL_BY_SCOPE]
        prompt = (
            f"Query: {query}\n"
            f"Requested evidence scope: {', '.join(scope)}\n"
            f"Allowed tools: {', '.join(allowed_tools)}"
        )
        try:
            structured = llm.complete_structured(
                [{"role": "user", "content": prompt}],
                schema_name="tool_plan",
            )
        except Exception:
            structured = None
        tool_plan = _valid_llm_steps(structured, scope)
        if tool_plan:
            return tool_plan
    rewritten = rewrite_for_scenario(query, scope)
    return [
        {
            "tool": TOOL_BY_SCOPE[item],
            "query": rewritten[item],
            "retrieval_mode": "hybrid_rerank",
            "required_evidence": item,
        }
        for item in scope
        if item in TOOL_BY_SCOPE
    ]
