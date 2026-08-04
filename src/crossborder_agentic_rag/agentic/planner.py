from .rewriter import rewrite_for_scenario


TOOL_BY_SCOPE = {
    "trademark": "trademark_search_tool",
    "patent": "patent_search_tool",
    "litigation": "litigation_search_tool",
}


def plan_tools(query: str, scope: list[str], llm=None) -> list[dict[str, object]]:
    if llm is not None:
        structured = llm.complete_structured(
            [{"role": "user", "content": query}],
            schema_name="tool_plan",
        )
        tool_plan = structured.get("tool_plan")
        if isinstance(tool_plan, list) and tool_plan:
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
