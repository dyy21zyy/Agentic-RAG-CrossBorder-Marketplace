"""LLM-driven Agentic RAG orchestration.

This module adds a true agentic retrieval loop on top of the existing
retrieval stack.

Core loop:
1. LLM planner decides required evidence and tool plan.
2. Python dispatcher executes planned tool actions.
3. Evidence evaluator checks whether evidence is sufficient.
4. LLM query rewriter creates follow-up queries for evidence gaps.
5. The loop stops when evidence is sufficient or max_iterations is reached.
6. A grounded answer is synthesized from retrieved evidence only.

The original rule-based AgenticRAG in graph.py is kept as a stable baseline.
"""
from __future__ import annotations

import json
import time
import os
from pathlib import Path

import re

from typing import Any

from crossborder_agentic_rag.agents.answer import synthesize_answer
from crossborder_agentic_rag.agents.classify import classify_query
from crossborder_agentic_rag.agents.evaluator import evaluate_evidence
from crossborder_agentic_rag.agents.llm_planner import LLMQueryPlan, ToolAction, plan_with_llm as _original_plan_with_llm
from crossborder_agentic_rag.agents.llm_query_rewriter import rewrite_with_llm
from crossborder_agentic_rag.agents.planner import build_query_plan
from crossborder_agentic_rag.agents.sql_router import SQLRouter
from crossborder_agentic_rag.retrieval.source_balanced import SourceBalancedRetriever
from crossborder_agentic_rag.retrieval.utils import dedupe_chunks
from crossborder_agentic_rag.schemas.queries import QueryPlan
from crossborder_agentic_rag.schemas.results import AgentState




def _write_llm_plan_probe_event(event: dict[str, Any]) -> None:
    path = os.environ.get("LLM_PLAN_PROBE_PATH", "/tmp/llm_plan_probe.jsonl")
    if not path:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        # probe must never break the main pipeline
        pass


def _llm_plan_action_to_dict(action: Any) -> dict[str, Any]:
    if isinstance(action, dict):
        return action
    out = {}
    for key in [
        "tool",
        "name",
        "query",
        "retrieval_mode",
        "required_evidence",
        "reason",
        "filters",
        "source_types",
    ]:
        try:
            value = getattr(action, key, None)
        except Exception:
            value = None
        if value is not None:
            out[key] = value
    if not out:
        out["repr"] = repr(action)
    return out


def plan_with_llm(*args: Any, **kwargs: Any) -> Any:
    """Probe wrapper around the real LLM planner.

    This verifies whether the Agent actually calls the LLM planning function
    and whether it returns a non-empty tool_plan.
    """
    started = time.time()

    query = kwargs.get("query")
    if not query:
        for x in args:
            if isinstance(x, str):
                query = x
                break
    query = query or ""

    _write_llm_plan_probe_event({
        "event": "llm_plan_attempted",
        "query": query,
        "ts": started,
        "args_types": [type(x).__name__ for x in args],
        "kwargs_keys": sorted(list(kwargs.keys())),
    })

    try:
        plan = _original_plan_with_llm(*args, **kwargs)

        if isinstance(plan, dict):
            tool_plan = plan.get("tool_plan") or []
        else:
            tool_plan = getattr(plan, "tool_plan", None) or []

        actions = [_llm_plan_action_to_dict(a) for a in tool_plan]
        tool_names = [
            a.get("tool") or a.get("name") or a.get("repr")
            for a in actions
        ]

        succeeded = bool(tool_plan)

        _write_llm_plan_probe_event({
            "event": "llm_plan_completed",
            "query": query,
            "succeeded": succeeded,
            "planner_source": "llm_function_called",
            "plan_type": type(plan).__name__,
            "tool_count": len(tool_plan),
            "tool_names": tool_names,
            "actions": actions,
            "elapsed_ms": round((time.time() - started) * 1000, 3),
        })

        return plan

    except Exception as exc:
        _write_llm_plan_probe_event({
            "event": "llm_plan_failed",
            "query": query,
            "succeeded": False,
            "planner_source": "llm_function_called_but_failed",
            "error": repr(exc),
            "elapsed_ms": round((time.time() - started) * 1000, 3),
        })
        raise


SOURCE_TYPES = {"trademark", "patent", "litigation"}

TOOL_TO_SOURCE = {
    "trademark_search_tool": "trademark",
    "patent_search_tool": "patent",
    "litigation_search_tool": "litigation",
}


class LLMAgenticRAG:
    """LLM-driven, rule-guarded, bounded-iteration Agentic RAG.

    The LLM is responsible for:
    - planning required evidence;
    - selecting tools;
    - rewriting follow-up queries.

    Python remains responsible for:
    - executing retrieval safely;
    - evidence sufficiency checking;
    - max iteration control;
    - grounded answer synthesis.
    """

    def __init__(
        self,
        duckdb_store: Any | None = None,
        retriever: Any | None = None,
        embedding_provider: Any | None = None,
        graph_retriever: Any | None = None,
        llm: Any | None = None,
        tools: list[Any] | None = None,
        max_iterations: int = 2,
        default_top_k: int = 20,
        retrieval_mode: str = "hybrid_rerank",
        candidate_k: int = 50,
        dense_k: int = 20,
        bm25_k: int = 20,
        rrf_k: int = 10,
    ) -> None:
        self.duckdb_store = duckdb_store
        self.retriever = retriever
        self.embedding_provider = embedding_provider
        self.graph_retriever = graph_retriever
        self.llm = llm
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.default_top_k = default_top_k
        self.retrieval_mode = retrieval_mode
        self.candidate_k = candidate_k
        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.rrf_k = rrf_k

    def _is_graph_query(self, query: str) -> bool:
        """Return True only for explicit relationship / multi-hop graph queries.

        Do not trigger GraphRAG for ordinary exact lookup queries such as:
        - Find trademark identity or registration evidence for XXX
        - Find patent claim evidence for patent XXX
        - Find litigation docket evidence for case XXX

        Those are better handled by DuckDB / BM25 / hybrid retrieval.
        """
        q = (query or "").lower()

        # Explicitly suppress GraphRAG for common exact lookup templates.
        exact_lookup_patterns = [
            "find trademark identity or registration evidence for",
            "find patent claim evidence for patent",
            "find litigation docket evidence for case",
            "which nice class evidence is associated with",
        ]
        if any(p in q for p in exact_lookup_patterns):
            return False

        graph_terms = [
            "graph",
            "connected",
            "connection",
            "relationship",
            "relationships",
            "linked",
            "link between",
            "related entities",
            "entity relationship",
            "multi-hop",
            "network",
            "associated with each other",
            "between",
        ]
        return any(t in q for t in graph_terms)

    def _ensure_graph_action(self, llm_plan, query: str) -> None:
        """Force graph_rag_tool for explicit graph/entity-relationship queries."""
        if not self._is_graph_query(query):
            return

        actions = getattr(llm_plan, "tool_plan", None)
        if actions is None:
            return

        for a in actions:
            if getattr(a, "tool", "") == "graph_rag_tool":
                return

        try:
            from crossborder_agentic_rag.agents.llm_planner import ToolAction
            actions.insert(
                0,
                ToolAction(
                    tool="graph_rag_tool",
                    query=query,
                    reason="Explicit graph/entity-relationship query; force NetworkX GraphRAG retrieval.",
                ),
            )
        except TypeError:
            # Some ToolAction versions may not accept reason.
            from crossborder_agentic_rag.agents.llm_planner import ToolAction
            actions.insert(
                0,
                ToolAction(
                    tool="graph_rag_tool",
                    query=query,
                ),
            )

    def _embed(self, query: str):
        if self.embedding_provider is None:
            return None
        return self.embedding_provider.embed_query(query)

    def _reranker_provider(self):
        r = getattr(self.retriever, "reranker", None) if self.retriever is not None else None
        if r is None:
            return None
        return getattr(r, "provider", None) or r.__class__.__name__

    def _payload(
        self,
        query: str,
        top_k: int,
        source_types: list[str] | None = None,
        required_evidence: str | None = None,
        reason: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "mode": self.retrieval_mode,
            "top_k": top_k,
            "candidate_k": self.candidate_k,
            "source_types": source_types,
        }
        if required_evidence:
            payload["required_evidence"] = required_evidence
        if reason:
            payload["reason"] = reason
        rp = self._reranker_provider()
        if rp:
            payload["reranker_provider"] = rp
        return payload

    def _rule_plan(self, query: str, top_k: int | None = None) -> QueryPlan:
        """Build a rule-based plan only as a compatibility fallback."""
        cls = classify_query(query)
        return build_query_plan(query, cls, top_k or self.default_top_k)

    def _expected_answer_type(self, llm_plan: LLMQueryPlan) -> str:
        query_type = (llm_plan.query_type or "").lower()
        required = set(llm_plan.required_evidence or [])

        if "mixed_ip_risk" in query_type or len(required & SOURCE_TYPES) > 1:
            return "risk_analysis"
        if "patent" in query_type or "patent" in required:
            return "patent_explanation"
        if "trademark" in query_type or "trademark" in required:
            return "trademark_explanation"
        if "litigation" in query_type or "litigation" in required:
            return "litigation_summary"
        if "structured" in query_type or "structured" in required:
            return "direct_field_answer"
        return "general_answer"

    def _retrieval_route(self, llm_plan: LLMQueryPlan, source_types: list[str]) -> str:
        required = set(llm_plan.required_evidence or [])
        if "structured" in required and not source_types:
            return "sql"
        if "structured" in required and source_types:
            return "mixed"
        if len(source_types) > 1:
            return "multi_source_risk"
        return "hybrid"

    def _build_query_plan_from_llm_plan(self, query: str, llm_plan: LLMQueryPlan) -> QueryPlan:
        source_types: list[str] = []
        for item in llm_plan.required_evidence or []:
            if item in SOURCE_TYPES and item not in source_types:
                source_types.append(item)

        if not source_types:
            for action in llm_plan.tool_plan:
                st = TOOL_TO_SOURCE.get(action.tool)
                if st and st not in source_types:
                    source_types.append(st)

        return QueryPlan(
            query=query,
            query_type=llm_plan.query_type or "general_ip_question",
            expected_answer_type=self._expected_answer_type(llm_plan),
            retrieval_route=self._retrieval_route(llm_plan, source_types),
            filters={},
            source_types=source_types,
            top_k=self.default_top_k,
        )

    def _retrieve(
        self,
        query: str,
        plan: QueryPlan,
        source_types: list[str] | None = None,
        top_k: int | None = None,
    ):
        if self.retriever is None:
            return []

        vec = self._embed(query)
        selected_top_k = top_k or plan.top_k
        st = source_types if source_types is not None else (plan.source_types or None)

        try:
            return self.retriever.retrieve(
                query,
                dense_vector=vec,
                filters=plan.filters,
                top_k=selected_top_k,
                source_types=st,
                mode=self.retrieval_mode,
                candidate_k=self.candidate_k,
            )
        except TypeError as exc:
            if "candidate_k" not in str(exc):
                raise
            return self.retriever.retrieve(
                query,
                dense_vector=vec,
                filters=plan.filters,
                top_k=selected_top_k,
                source_types=st,
                mode=self.retrieval_mode,
            )

    def _source_balanced_retrieve(self, query: str, plan: QueryPlan, source_types: list[str]):
        if self.retriever is None:
            return []

        balanced = SourceBalancedRetriever(
            self.retriever,
            mode=self.retrieval_mode,
            per_source_k=min(plan.top_k, max(1, self.candidate_k)),
            final_k=plan.top_k,
            candidate_k=self.candidate_k,
        )

        return balanced.retrieve(
            query,
            dense_vector=self._embed(query),
            filters=plan.filters,
            source_types=source_types,
        )

    def _run_search_action(
        self,
        state: AgentState,
        action: ToolAction,
        plan: QueryPlan,
    ) -> None:
        source_type = TOOL_TO_SOURCE.get(action.tool)

        if source_type is None:
            state.add_trace("skip_unknown_search_tool")
            state.add_tool_call(
                action.tool,
                {
                    "query": action.query,
                    "error": "Unknown source search tool.",
                },
            )
            return

        if self.retriever is None:
            state.add_trace("retriever_unavailable")
            state.add_tool_call(
                action.tool,
                {
                    "query": action.query,
                    "source_types": [source_type],
                    "error": "Retriever is not configured.",
                },
            )
            return

        evidence = self._retrieve(action.query, plan, source_types=[source_type])
        state.retrieved_evidence = dedupe_chunks([*state.retrieved_evidence, *evidence])

        state.add_tool_call(
            action.tool,
            self._payload(
                query=action.query,
                top_k=plan.top_k,
                source_types=[source_type],
                required_evidence=action.required_evidence,
                reason=action.reason,
            ),
        )
        state.add_trace("tool_search_retrieval")

    def _run_duckdb_action(
        self,
        state: AgentState,
        action: ToolAction,
    ) -> None:
        if self.duckdb_store is None:
            state.add_trace("duckdb_unavailable")
            state.add_tool_call(
                "duckdb_lookup_tool",
                {
                    "query": action.query,
                    "error": "DuckDB store is not configured.",
                },
            )
            return

        sql_plan = self._rule_plan(action.query, top_k=self.default_top_k)
        rows = SQLRouter(self.duckdb_store).run(sql_plan)

        state.sql_results.extend(rows or [])
        state.add_tool_call(
            "duckdb_lookup_tool",
            {
                "query": action.query,
                "filters": sql_plan.filters,
                "rows": len(rows or []),
                "reason": action.reason,
            },
        )
        state.add_trace("tool_duckdb_lookup")

    def _chunk_id_of(self, chunk):
        """Get chunk_id from EvidenceChunk-like object or dict."""
        if chunk is None:
            return None
        if isinstance(chunk, dict):
            return chunk.get("chunk_id") or chunk.get("id")
        return getattr(chunk, "chunk_id", None) or getattr(chunk, "id", None)

    def _all_available_chunks(self):
        """Collect chunks from retriever/BM25 objects for graph chunk_id lookup."""
        chunks = []

        candidates = [
            self.retriever,
            getattr(self.retriever, "bm25", None) if self.retriever is not None else None,
            getattr(self.retriever, "bm25_retriever", None) if self.retriever is not None else None,
            getattr(self.retriever, "sparse_retriever", None) if self.retriever is not None else None,
        ]

        attr_names = [
            "chunks",
            "_chunks",
            "documents",
            "_documents",
            "docs",
            "_docs",
            "corpus",
            "_corpus",
        ]

        for obj in candidates:
            if obj is None:
                continue
            for attr in attr_names:
                val = getattr(obj, attr, None)
                if isinstance(val, list) and val:
                    chunks.extend(val)

        # 去重
        out = []
        seen = set()
        for c in chunks:
            cid = self._chunk_id_of(c)
            key = cid or id(c)
            if key in seen:
                continue
            seen.add(key)
            out.append(c)

        return out

    def _lookup_chunks_by_ids(self, chunk_ids, limit: int = 50):
        """Look up EvidenceChunk objects by chunk_ids."""
        wanted = []
        seen = set()

        for cid in chunk_ids or []:
            if not cid or cid in seen:
                continue
            wanted.append(cid)
            seen.add(cid)

        if not wanted:
            return []

        wanted_set = set(wanted)
        found = []

        for chunk in self._all_available_chunks():
            cid = self._chunk_id_of(chunk)
            if cid in wanted_set:
                found.append(chunk)
                if len(found) >= limit:
                    break

        return found

    def _merge_evidence_into_state(self, state, new_evidence):
        """Merge new evidence chunks into state.retrieved_evidence with chunk_id dedupe."""
        if not new_evidence:
            return 0

        existing = list(getattr(state, "retrieved_evidence", []) or [])
        seen = set()

        for c in existing:
            cid = self._chunk_id_of(c)
            if cid:
                seen.add(cid)

        added = 0
        for c in new_evidence:
            cid = self._chunk_id_of(c)
            if cid and cid in seen:
                continue
            existing.append(c)
            if cid:
                seen.add(cid)
            added += 1

        state.retrieved_evidence = existing
        return added

    def _run_graph_action(self, state, action):
        """Run NetworkX GraphRAG tool, log graph stats, and merge graph chunks into evidence."""
        if self.graph_retriever is None:
            state.add_tool_call(
                "graph_rag_tool",
                {
                    "query": getattr(action, "query", ""),
                    "status": "unavailable",
                    "reason": "graph_retriever is None",
                },
            )
            state.add_trace("graph_unavailable")
            return None

        query = getattr(action, "query", "") or ""
        result = self.graph_retriever.retrieve(query)

        related_nodes = result.get("related_nodes", []) if isinstance(result, dict) else []
        related_edges = result.get("related_edges", []) if isinstance(result, dict) else []
        related_chunk_ids = result.get("related_chunk_ids", []) if isinstance(result, dict) else []
        matched_entities = result.get("matched_entities", []) if isinstance(result, dict) else []

        # 把 graph related_chunk_ids 回查成 EvidenceChunk，并合并进 retrieved_evidence
        graph_evidence = self._lookup_chunks_by_ids(
            related_chunk_ids,
            limit=max(self.default_top_k, 20),
        )
        graph_evidence_added = self._merge_evidence_into_state(state, graph_evidence)

        state.add_tool_call(
            "graph_rag_tool",
            {
                "query": query,
                "status": "ok",
                "matched_entities": len(matched_entities),
                "related_nodes": len(related_nodes),
                "related_edges": len(related_edges),
                "related_chunk_ids": len(related_chunk_ids),
                "graph_evidence_found": len(graph_evidence),
                "graph_evidence_added": graph_evidence_added,
                "count": len(related_chunk_ids),
                "reason": getattr(action, "reason", None),
            },
        )

        try:
            state.graph_result = result
        except Exception:
            pass

        state.add_trace("tool_graph_retrieval")
        return result


    def _run_tool_action(
        self,
        state: AgentState,
        action: ToolAction,
        plan: QueryPlan,
    ) -> None:
        if action.tool in TOOL_TO_SOURCE:
            self._run_search_action(state, action, plan)
        elif action.tool == "duckdb_lookup_tool":
            self._run_duckdb_action(state, action)
        elif action.tool == "graph_rag_tool":
            self._run_graph_action(state, action)
        else:
            state.add_trace("unknown_tool_action")
            state.add_tool_call(
                action.tool,
                {
                    "query": action.query,
                    "error": "Tool not supported by LLMAgenticRAG dispatcher.",
                },
            )

    def _run_initial_plan(
        self,
        state: AgentState,
        plan: QueryPlan,
        llm_plan: LLMQueryPlan,
    ) -> None:
        if len(plan.source_types) > 1 and self.retriever is not None:
            search_actions = [a for a in llm_plan.tool_plan if a.tool in TOOL_TO_SOURCE]
            planned_sources = [TOOL_TO_SOURCE[a.tool] for a in search_actions if a.tool in TOOL_TO_SOURCE]
            if set(planned_sources) == set(plan.source_types):
                evidence = self._source_balanced_retrieve(plan.query, plan, plan.source_types)
                state.retrieved_evidence = dedupe_chunks([*state.retrieved_evidence, *evidence])
                state.add_tool_call(
                    "source_balanced_retriever",
                    self._payload(plan.query, plan.top_k, plan.source_types),
                )
                state.add_trace("llm_plan_source_balanced_retrieval")

                remaining_actions = [a for a in llm_plan.tool_plan if a.tool not in TOOL_TO_SOURCE]
                for action in remaining_actions:
                    self._run_tool_action(state, action, plan)
                return

        for action in llm_plan.tool_plan:
            self._run_tool_action(state, action, plan)

    def _finalize_evidence(self, query: str, evidence, top_k: int):
        deduped = dedupe_chunks(evidence)
        reranker = getattr(self.retriever, "reranker", None) if self.retriever is not None else None

        if self.retrieval_mode == "hybrid_rerank" and reranker is not None:
            reranked = dedupe_chunks(reranker.rerank(query, deduped, top_k))[:top_k]
            final_call = {
                "tool": "final_rerank",
                "payload": {
                    "reranker_provider": self._reranker_provider(),
                    "input_evidence_count": len(deduped),
                    "output_evidence_count": min(len(deduped), top_k),
                },
            }
            return deduped, reranked, final_call

        return deduped, deduped[:top_k], None

    def run(self, query: str) -> AgentState:
        state = AgentState(query=query)
        state.retrieval_mode = self.retrieval_mode
        state.candidate_k = self.candidate_k
        state.reranker_provider = self._reranker_provider()

        available_tool_names = [
            "trademark_search_tool",
            "patent_search_tool",
            "litigation_search_tool",
            "duckdb_lookup_tool",
            "graph_rag_tool",
        ]

        llm_plan = plan_with_llm(
            query=query,
            llm=self.llm,
            tools=self.tools,
            available_tool_names=available_tool_names,
            max_tools=5,
        )
        state.add_trace("llm_plan")
        self._ensure_graph_action(llm_plan, query)
        state.normalized_query = query
        state.query_type = llm_plan.query_type
        state.retrieval_route = "agentic_llm"

        plan = self._build_query_plan_from_llm_plan(query, llm_plan)
        state.expected_answer_type = plan.expected_answer_type
        state.retrieval_plan.append(plan)
        state.add_trace("build_llm_query_plan")

        self._run_initial_plan(state, plan, llm_plan)

        # FORCE_INITIAL_TOOL_PLAN_BEFORE_EVALUATION
        # Execute the LLM-selected tool plan before evidence evaluation.
        # Otherwise the evaluator sees zero evidence and only triggers follow-up retrieval.
        if not getattr(state, "retrieved_evidence", None):
            self._run_initial_plan(llm_plan, plan, state)


        ev = evaluate_evidence(plan, state.retrieved_evidence, state.sql_results)
        state.evidence_gaps = ev.evidence_gaps
        state.add_trace("evaluate_evidence")

        while not ev.is_sufficient and state.iterations < self.max_iterations:
            state.iterations += 1

            rewrite_plan = rewrite_with_llm(
                original_query=query,
                evidence_gaps=ev.evidence_gaps,
                current_evidence=state.retrieved_evidence,
                llm=self.llm,
                available_tool_names=available_tool_names,
                max_queries=5,
            )
            state.add_trace("llm_query_rewrite")

            if not rewrite_plan.followup_queries:
                state.add_trace("no_followup_queries")
                break

            for followup in rewrite_plan.followup_queries:
                action = ToolAction(
                    tool=followup.tool,
                    query=followup.query,
                    reason=followup.reason,
                    required_evidence=followup.missing_evidence,
                )
                self._run_tool_action(state, action, plan)
                state.add_trace("followup_tool_retrieval")

            ev = evaluate_evidence(plan, state.retrieved_evidence, state.sql_results)
            state.evidence_gaps = ev.evidence_gaps
            state.add_trace("evaluate_evidence")

        state.retrieved_evidence, state.reranked_evidence, final_call = self._finalize_evidence(
            query=plan.query,
            evidence=state.retrieved_evidence,
            top_k=plan.top_k,
        )

        if final_call is not None:
            state.add_tool_call(final_call["tool"], final_call["payload"])
            state.add_trace("final_rerank")

        evidence = state.reranked_evidence or dedupe_chunks(state.retrieved_evidence)
        state.answer, state.citations = synthesize_answer(
            plan,
            state.sql_results,
            evidence,
            state.evidence_gaps,
        )
        state.add_trace("final_answer")

        return state

    # ===== BEGIN adaptive per-action retrieval-mode overrides =====

    def _retrieve(
        self,
        query,
        plan,
        source_types=None,
        top_k=None,
        retrieval_mode=None,
    ):
        """Retrieve evidence with the retrieval mode selected for this tool action.

        This override makes true Agentic RAG execute:
        - trademark_search_tool with its own retrieval_mode;
        - patent_search_tool with its own retrieval_mode;
        - litigation_search_tool with its own retrieval_mode.

        The global self.retrieval_mode is only used as fallback.
        """
        from crossborder_agentic_rag.agents.retrieval_policy import (
            DEFAULT_RETRIEVAL_MODE,
            normalize_retrieval_mode,
        )

        if self.retriever is None:
            return []

        selected_mode = normalize_retrieval_mode(
            retrieval_mode or getattr(self, "retrieval_mode", None),
            fallback=DEFAULT_RETRIEVAL_MODE,
        )

        k = top_k or getattr(self, "default_top_k", 20)
        candidate_k = getattr(self, "candidate_k", None)

        # Different retriever implementations may use slightly different signatures.
        # Try the richest signature first, then fall back safely.
        attempts = [
            lambda: self.retriever.retrieve(
                query=query,
                top_k=k,
                mode=selected_mode,
                source_types=source_types,
                candidate_k=candidate_k,
                dense_k=getattr(self, "dense_k", 20),
                bm25_k=getattr(self, "bm25_k", 20),
                rrf_k=getattr(self, "rrf_k", 10),
            ),
            lambda: self.retriever.retrieve(
                query=query,
                top_k=k,
                mode=selected_mode,
                source_types=source_types,
            ),
            lambda: self.retriever.retrieve(
                query,
                top_k=k,
                mode=selected_mode,
                source_types=source_types,
            ),
            lambda: self.retriever.retrieve(
                query=query,
                top_k=k,
                retrieval_mode=selected_mode,
                source_types=source_types,
            ),
            lambda: self.retriever.retrieve(query, k, selected_mode, source_types),
        ]

        last_error = None
        for fn in attempts:
            try:
                items = fn()
                return list(items or [])
            except TypeError as exc:
                last_error = exc
                continue

        raise last_error

    def _run_search_action(self, action, plan, state):
        """Run trademark / patent / litigation search action with action-level retrieval mode."""
        from crossborder_agentic_rag.agents.retrieval_policy import (
            DEFAULT_RETRIEVAL_MODE,
            normalize_retrieval_mode,
        )

        tool_to_source = {
            "trademark_search_tool": "trademark",
            "patent_search_tool": "patent",
            "litigation_search_tool": "litigation",
        }

        source_type = tool_to_source.get(getattr(action, "tool", ""))
        if source_type is None:
            state.add_trace(f"unknown_search_tool:{getattr(action, 'tool', '')}")
            return []

        selected_mode = normalize_retrieval_mode(
            getattr(action, "retrieval_mode", None) or getattr(self, "retrieval_mode", None),
            fallback=DEFAULT_RETRIEVAL_MODE,
        )

        action_query = getattr(action, "query", "") or ""

        # Guardrail for exact patent-id queries:
        # dense_only is unsafe when the query contains a specific patent number,
        # because vector search may retrieve semantically similar claims from another patent.
        # Keep the LLM-selected tool, but switch the retrieval backend to hybrid_rerank
        # so BM25 exact matching can be triggered.
        if (
            getattr(action, "tool", "") == "patent_search_tool"
            and selected_mode == "dense_only"
            and re.search(r"\bpatent\s+\d{7,12}\b", action_query, flags=re.I)
        ):
            selected_mode = "hybrid_rerank"

        evidence = self._retrieve(
            action_query,
            plan,
            source_types=[source_type],
            retrieval_mode=selected_mode,
        )

        if hasattr(state, "retrieved_evidence") and isinstance(state.retrieved_evidence, list):
            state.retrieved_evidence.extend(evidence)

        if hasattr(state, "add_tool_call"):
            state.add_tool_call(
                getattr(action, "tool", ""),
                {
                    "query": getattr(action, "query", ""),
                    "source_types": [source_type],
                    "retrieval_mode": selected_mode,
                    "count": len(evidence),
                    "reason": getattr(action, "reason", ""),
                    "required_evidence": getattr(action, "required_evidence", None),
                },
            )

        if hasattr(state, "add_trace"):
            state.add_trace("tool_retrieval")

        return evidence

    def _run_tool_action(self, action, plan, state):
        """Dispatch one planned tool action."""
        tool = getattr(action, "tool", "")

        if tool in {"trademark_search_tool", "patent_search_tool", "litigation_search_tool"}:
            return self._run_search_action(action, plan, state)

        if tool == "duckdb_lookup_tool":
            return self._run_duckdb_action(state, action)

        if tool == "graph_rag_tool":
            return self._run_graph_action(state, action)

        if hasattr(state, "add_trace"):
            state.add_trace(f"unknown_tool:{tool}")
        return []

    def _run_initial_plan(self, llm_plan, plan, state):
        """Run the LLM-planned tool actions one by one.

        Important:
        We intentionally do NOT collapse multiple sources into one source-balanced retrieval here,
        because each action may have its own retrieval_mode.
        """
        if hasattr(state, "add_trace"):
            state.add_trace("run_initial_tool_plan")

        for action in getattr(llm_plan, "tool_plan", []) or []:
            self._run_tool_action(action, plan, state)

    # ===== END adaptive per-action retrieval-mode overrides =====

