"""Rule-based adaptive tool-routing and retrieval-mode policy.

This module is not only choosing a retrieval mode.
It builds a deterministic tool plan:

query
-> query type
-> evidence source tools
-> retrieval mode for each tool

Example:
- trademark risk:
    trademark_search_tool + hybrid_rrf / hybrid_rerank
- patent claim risk:
    patent_search_tool + dense_only
- litigation exact docket:
    duckdb_lookup_tool + litigation_search_tool + bm25_only
- entity relation:
    graph_rag_tool + hybrid_rrf
- mixed IP risk:
    trademark + patent + litigation tool combination
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re


VALID_RETRIEVAL_MODES = {
    "bm25_only",
    "dense_only",
    "hybrid_rrf",
    "hybrid_rerank",
}

DEFAULT_RETRIEVAL_MODE = "hybrid_rerank"


SEARCH_TOOLS = {
    "trademark": "trademark_search_tool",
    "patent": "patent_search_tool",
    "litigation": "litigation_search_tool",
    "structured": "duckdb_lookup_tool",
    "graph": "graph_rag_tool",
}


@dataclass(slots=True)
class RuleToolAction:
    tool: str
    query: str
    retrieval_mode: str | None = None
    source_type: str | None = None
    reason: str = ""
    priority: int = 1


@dataclass(slots=True)
class RuleToolPlan:
    query: str
    query_type: str
    expected_answer_type: str
    actions: list[RuleToolAction] = field(default_factory=list)
    reason: str = ""


EXACT_IDENTIFIER_PATTERNS = [
    r"\b[0-9]+:[0-9]{2}-[a-z]{2}-[0-9]+\b",          # 3:90-cv-04129
    r"\bcase\s*(number|no\.?|#)?\s*[:#]?\s*[0-9]+:[0-9]{2}-[a-z]{2}-[0-9]+\b",
    r"\bdocket\s*(number|no\.?|#)\b",
    r"\bregistration\s*(number|no\.?|#)\b",
    r"\bserial\s*(number|no\.?|#)\b",
    r"\bpatent\s*(number|no\.?|id|#)\b",
    r"\bUS\s*[0-9][0-9A-Z,/-]{3,}\b",
    r"\bword\s+mark\b",
    r"\bnice\s+class\b",
]


def normalize_retrieval_mode(value: str | None, fallback: str = DEFAULT_RETRIEVAL_MODE) -> str:
    mode = (value or "").strip().lower()
    if mode in VALID_RETRIEVAL_MODES:
        return mode
    return fallback


def has_any(query: str, terms: list[str]) -> bool:
    q = (query or "").lower()
    return any(t in q for t in terms)


def is_exact_identifier_query(query: str) -> bool:
    q = query or ""
    return any(re.search(p, q, re.I) for p in EXACT_IDENTIFIER_PATTERNS)


def is_litigation_query(query: str) -> bool:
    return has_any(query, [
        "litigation", "lawsuit", "case", "docket", "plaintiff", "defendant",
        "court", "complaint", "sued", "asserted patent", "patent infringement lawsuit",
        "cv-",
    ]) or bool(re.search(r"\b[0-9]+:[0-9]{2}-[a-z]{2}-[0-9]+\b", query or "", re.I))


def is_patent_query(query: str) -> bool:
    return has_any(query, [
        "patent", "claim", "claims", "invention", "utility", "design patent",
        "technical feature", "foldable", "mechanism", "device", "apparatus",
        "infringe a patent", "patent infringement",
    ])


def is_trademark_query(query: str) -> bool:
    return has_any(query, [
        "trademark", "word mark", "brand", "logo", "mark", "nice class",
        "goods", "services", "counterfeit", "confusingly similar",
        "backpack", "travel bags", "t-shirt", "nike-like",
    ])


def is_graph_query(query: str) -> bool:
    return has_any(query, [
        "relationship", "relation", "connected", "connection", "multi-hop",
        "entity", "entities", "network", "between", "associated with",
        "owner", "assignee", "plaintiff", "defendant", "asserted patent",
        "which company", "who sued whom", "related cases",
    ])


def is_mixed_risk_query(query: str) -> bool:
    q = (query or "").lower()

    risk_terms = [
        "risk", "risky", "can i sell", "should i list", "safe to sell",
        "infringe", "infringement", "counterfeit", "ip risk",
        "compliance", "marketplace",
    ]

    source_hits = 0
    if is_trademark_query(q):
        source_hits += 1
    if is_patent_query(q):
        source_hits += 1
    if is_litigation_query(q):
        source_hits += 1

    return has_any(q, risk_terms) and source_hits >= 2


def explain_retrieval_mode(mode: str | None) -> str:
    mode = normalize_retrieval_mode(mode)

    if mode == "bm25_only":
        return "BM25 lexical retrieval, suitable for exact identifiers, docket numbers, registration numbers, exact word marks, and keyword-heavy lookup."
    if mode == "dense_only":
        return "Dense semantic retrieval, suitable for patent claims, product features, and paraphrased technical descriptions."
    if mode == "hybrid_rrf":
        return "BM25 + dense fusion with RRF, suitable when both keyword matching and semantic recall are useful."
    if mode == "hybrid_rerank":
        return "Hybrid retrieval followed by reranking, suitable for high-precision evidence, mixed IP risk, semantic litigation, and final decision support."

    return "Unknown retrieval mode."


def choose_rule_retrieval_mode(
    query: str,
    source_type: str | None = None,
    expected_answer_type: str | None = None,
) -> str:
    """Backward-compatible function: choose one mode for one source.

    This is kept so old code will not break.
    The new preferred function is build_rule_tool_plan().
    """
    q = query or ""
    st = (source_type or "").lower()
    ans = (expected_answer_type or "").lower()

    if is_exact_identifier_query(q):
        return "bm25_only"

    if st == "structured":
        return "bm25_only"

    if st == "patent" or "patent" in ans:
        if has_any(q, ["claim", "technical feature", "mechanism", "device", "apparatus"]):
            return "dense_only"
        return "hybrid_rerank"

    if st == "litigation" or "litigation" in ans:
        if is_exact_identifier_query(q) or has_any(q, ["case number", "docket number", "cv-"]):
            return "bm25_only"
        return "hybrid_rerank"

    if st == "trademark" or "trademark" in ans:
        if has_any(q, ["exact word mark", "registration number", "serial number", "nice class"]):
            return "bm25_only"
        if has_any(q, ["similar", "logo", "counterfeit", "risk", "can i sell", "infringe"]):
            return "hybrid_rerank"
        return "hybrid_rrf"

    if is_graph_query(q):
        return "hybrid_rrf"

    if is_mixed_risk_query(q):
        return "hybrid_rerank"

    return DEFAULT_RETRIEVAL_MODE


def choose_followup_retrieval_mode(missing_evidence: str, query: str) -> str:
    missing = (missing_evidence or "").lower().replace("missing ", "").replace(" evidence", "").strip()

    if missing == "structured":
        return "bm25_only"
    if missing == "trademark":
        return choose_rule_retrieval_mode(query, source_type="trademark")
    if missing == "patent":
        return choose_rule_retrieval_mode(query, source_type="patent")
    if missing == "litigation":
        return choose_rule_retrieval_mode(query, source_type="litigation")
    if missing == "graph":
        return "hybrid_rrf"

    return DEFAULT_RETRIEVAL_MODE


def _add_action(
    actions: list[RuleToolAction],
    tool: str,
    query: str,
    retrieval_mode: str | None,
    source_type: str | None,
    reason: str,
    priority: int,
) -> None:
    """Add an action and avoid exact duplicates."""
    for a in actions:
        if a.tool == tool and a.query == query and a.retrieval_mode == retrieval_mode and a.source_type == source_type:
            return

    actions.append(
        RuleToolAction(
            tool=tool,
            query=query,
            retrieval_mode=retrieval_mode,
            source_type=source_type,
            reason=reason,
            priority=priority,
        )
    )


def build_rule_tool_plan(query: str) -> RuleToolPlan:
    """Build deterministic rule-based adaptive retrieval plan.

    This is the main rule-based planner.

    It chooses:
    - which tool to call;
    - which retrieval mode each search tool should use;
    - whether DuckDB or GraphRAG should be triggered.
    """
    q = query or ""
    q_lower = q.lower()
    actions: list[RuleToolAction] = []

    # 1. Exact litigation / docket / case lookup:
    #    DuckDB first, then BM25 litigation evidence.
    if is_litigation_query(q) and is_exact_identifier_query(q):
        _add_action(
            actions,
            tool="duckdb_lookup_tool",
            query=q,
            retrieval_mode=None,
            source_type="structured",
            reason="Exact litigation identifier detected; use DuckDB for structured lookup.",
            priority=1,
        )
        _add_action(
            actions,
            tool="litigation_search_tool",
            query=q,
            retrieval_mode="bm25_only",
            source_type="litigation",
            reason="Exact litigation case or docket query; BM25 is strong for exact case identifiers.",
            priority=2,
        )
        return RuleToolPlan(
            query=q,
            query_type="litigation_exact_lookup",
            expected_answer_type="litigation_answer",
            actions=sorted(actions, key=lambda x: x.priority),
            reason="Rule-based exact litigation lookup: DuckDB + BM25 litigation evidence.",
        )

    # 2. Exact patent number / registration / serial lookup:
    #    DuckDB + BM25 in the likely source.
    if is_exact_identifier_query(q) and is_patent_query(q):
        _add_action(
            actions,
            tool="duckdb_lookup_tool",
            query=q,
            retrieval_mode=None,
            source_type="structured",
            reason="Exact patent identifier detected; use DuckDB for structured lookup.",
            priority=1,
        )
        _add_action(
            actions,
            tool="patent_search_tool",
            query=q,
            retrieval_mode="bm25_only",
            source_type="patent",
            reason="Exact patent identifier query; BM25 can match patent numbers and claim identifiers.",
            priority=2,
        )
        return RuleToolPlan(
            query=q,
            query_type="patent_exact_lookup",
            expected_answer_type="patent_answer",
            actions=sorted(actions, key=lambda x: x.priority),
            reason="Rule-based exact patent lookup: DuckDB + BM25 patent evidence.",
        )

    if is_exact_identifier_query(q) and is_trademark_query(q):
        _add_action(
            actions,
            tool="duckdb_lookup_tool",
            query=q,
            retrieval_mode=None,
            source_type="structured",
            reason="Exact trademark identifier or word mark detected; use DuckDB for structured lookup.",
            priority=1,
        )
        _add_action(
            actions,
            tool="trademark_search_tool",
            query=q,
            retrieval_mode="bm25_only",
            source_type="trademark",
            reason="Exact trademark query; BM25 is suitable for word marks, serial numbers, and Nice classes.",
            priority=2,
        )
        return RuleToolPlan(
            query=q,
            query_type="trademark_exact_lookup",
            expected_answer_type="trademark_answer",
            actions=sorted(actions, key=lambda x: x.priority),
            reason="Rule-based exact trademark lookup: DuckDB + BM25 trademark evidence.",
        )

    # 3. Mixed IP risk:
    #    query needs trademark + patent + litigation evidence.
    if is_mixed_risk_query(q):
        _add_action(
            actions,
            tool="trademark_search_tool",
            query=q,
            retrieval_mode="hybrid_rrf",
            source_type="trademark",
            reason="Mixed IP risk query; trademark evidence needs both lexical brand terms and semantic goods overlap.",
            priority=1,
        )
        _add_action(
            actions,
            tool="patent_search_tool",
            query=q,
            retrieval_mode="dense_only",
            source_type="patent",
            reason="Mixed IP risk query; patent claim risk needs semantic matching over claims and product features.",
            priority=2,
        )
        _add_action(
            actions,
            tool="litigation_search_tool",
            query=q,
            retrieval_mode="hybrid_rerank",
            source_type="litigation",
            reason="Mixed IP risk query; litigation evidence benefits from hybrid retrieval and reranking.",
            priority=3,
        )
        if is_graph_query(q):
            _add_action(
                actions,
                tool="graph_rag_tool",
                query=q,
                retrieval_mode="hybrid_rrf",
                source_type="graph",
                reason="Complex entity relation detected; trigger GraphRAG for multi-hop evidence.",
                priority=4,
            )
        return RuleToolPlan(
            query=q,
            query_type="mixed_ip_risk",
            expected_answer_type="risk_answer",
            actions=sorted(actions, key=lambda x: x.priority),
            reason="Rule-based mixed-risk plan: trademark + patent + litigation evidence.",
        )

    # 4. Trademark risk / logo / brand similarity:
    if is_trademark_query(q):
        mode = "hybrid_rerank" if has_any(q_lower, ["risk", "risky", "similar", "logo", "counterfeit", "can i sell", "infringe"]) else "hybrid_rrf"
        _add_action(
            actions,
            tool="trademark_search_tool",
            query=q,
            retrieval_mode=mode,
            source_type="trademark",
            reason="Trademark query; use hybrid retrieval for mark similarity, goods/services overlap, and brand-risk evidence.",
            priority=1,
        )
        return RuleToolPlan(
            query=q,
            query_type="trademark",
            expected_answer_type="trademark_answer",
            actions=sorted(actions, key=lambda x: x.priority),
            reason="Rule-based trademark plan.",
        )

    # 5. Patent claim / technical feature:
    if is_patent_query(q):
        mode = "dense_only" if has_any(q_lower, ["claim", "technical feature", "mechanism", "device", "apparatus", "foldable"]) else "hybrid_rerank"
        _add_action(
            actions,
            tool="patent_search_tool",
            query=q,
            retrieval_mode=mode,
            source_type="patent",
            reason="Patent query; use dense retrieval for semantic claim/product-feature matching.",
            priority=1,
        )
        return RuleToolPlan(
            query=q,
            query_type="patent",
            expected_answer_type="patent_answer",
            actions=sorted(actions, key=lambda x: x.priority),
            reason="Rule-based patent claim / technical evidence plan.",
        )

    # 6. Litigation semantic query:
    if is_litigation_query(q):
        _add_action(
            actions,
            tool="litigation_search_tool",
            query=q,
            retrieval_mode="hybrid_rerank",
            source_type="litigation",
            reason="Semantic litigation query; use hybrid retrieval and reranking instead of pure BM25.",
            priority=1,
        )
        if is_graph_query(q):
            _add_action(
                actions,
                tool="graph_rag_tool",
                query=q,
                retrieval_mode="hybrid_rrf",
                source_type="graph",
                reason="Litigation query has entity-relation intent; trigger GraphRAG.",
                priority=2,
            )
        return RuleToolPlan(
            query=q,
            query_type="litigation",
            expected_answer_type="litigation_answer",
            actions=sorted(actions, key=lambda x: x.priority),
            reason="Rule-based litigation semantic retrieval plan.",
        )

    # 7. Pure graph / entity relationship query:
    if is_graph_query(q):
        _add_action(
            actions,
            tool="graph_rag_tool",
            query=q,
            retrieval_mode="hybrid_rrf",
            source_type="graph",
            reason="Complex entity relationship query; GraphRAG is triggered.",
            priority=1,
        )
        return RuleToolPlan(
            query=q,
            query_type="entity_relation",
            expected_answer_type="graph_answer",
            actions=sorted(actions, key=lambda x: x.priority),
            reason="Rule-based GraphRAG plan.",
        )

    # 8. Fallback:
    #    Use hybrid_rerank over all main evidence sources for robust answer.
    _add_action(
        actions,
        tool="trademark_search_tool",
        query=q,
        retrieval_mode="hybrid_rrf",
        source_type="trademark",
        reason="Fallback query; retrieve possible trademark evidence.",
        priority=1,
    )
    _add_action(
        actions,
        tool="patent_search_tool",
        query=q,
        retrieval_mode="dense_only",
        source_type="patent",
        reason="Fallback query; retrieve possible patent evidence semantically.",
        priority=2,
    )
    _add_action(
        actions,
        tool="litigation_search_tool",
        query=q,
        retrieval_mode="hybrid_rerank",
        source_type="litigation",
        reason="Fallback query; retrieve possible litigation evidence.",
        priority=3,
    )

    return RuleToolPlan(
        query=q,
        query_type="general_ip_question",
        expected_answer_type="risk_answer",
        actions=sorted(actions, key=lambda x: x.priority),
        reason="Fallback rule-based multi-source IP evidence plan.",
    )


def plan_to_dict(plan: RuleToolPlan) -> dict:
    return {
        "query": plan.query,
        "query_type": plan.query_type,
        "expected_answer_type": plan.expected_answer_type,
        "reason": plan.reason,
        "actions": [
            {
                "tool": a.tool,
                "query": a.query,
                "retrieval_mode": a.retrieval_mode,
                "source_type": a.source_type,
                "reason": a.reason,
                "priority": a.priority,
            }
            for a in plan.actions
        ],
    }
