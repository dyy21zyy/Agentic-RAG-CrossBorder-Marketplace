"""Agentic RAG orchestration graph for Stage 6."""
from __future__ import annotations
from typing import Any
from crossborder_agentic_rag.agents.answer import synthesize_answer
from crossborder_agentic_rag.agents.classify import classify_query
from crossborder_agentic_rag.agents.evaluator import evaluate_evidence
from crossborder_agentic_rag.agents.planner import build_query_plan
from crossborder_agentic_rag.agents.sql_router import SQLRouter
from crossborder_agentic_rag.retrieval.source_balanced import SourceBalancedRetriever
from crossborder_agentic_rag.retrieval.utils import dedupe_chunks
from crossborder_agentic_rag.schemas.results import AgentState

class AgenticRAG:
    def __init__(self, duckdb_store: Any | None = None, retriever: Any | None = None, embedding_provider: Any | None = None, max_iterations: int = 2, default_top_k: int = 20, retrieval_mode: str = "hybrid_rrf", candidate_k: int = 50) -> None:
        self.duckdb_store=duckdb_store; self.retriever=retriever; self.embedding_provider=embedding_provider; self.max_iterations=max_iterations; self.default_top_k=default_top_k; self.retrieval_mode=retrieval_mode; self.candidate_k=candidate_k
    def _embed(self, query: str): return None if self.embedding_provider is None else self.embedding_provider.embed_query(query)
    def _reranker_provider(self):
        r=getattr(self.retriever,"reranker",None)
        if r is None: return None
        return getattr(r,"provider",None) or r.__class__.__name__
    def _payload(self, top_k, source_types=None, followup_query=None):
        payload={"mode":self.retrieval_mode,"top_k":top_k,"candidate_k":self.candidate_k,"source_types":source_types}
        rp=self._reranker_provider()
        if rp: payload["reranker_provider"]=rp
        if followup_query: payload["followup_query"]=followup_query
        return payload
    def _retrieve(self, query, plan, source_types=None, mode: str | None = None):
        if self.retriever is None: raise ValueError("Retriever is required for hybrid retrieval")
        vec=self._embed(query); selected_mode=mode or self.retrieval_mode
        st=source_types if source_types is not None else (plan.source_types or None)
        try:
            return self.retriever.retrieve(query, dense_vector=vec, filters=plan.filters, top_k=plan.top_k, source_types=st, mode=selected_mode, candidate_k=self.candidate_k)
        except TypeError as exc:
            if "candidate_k" not in str(exc): raise
            return self.retriever.retrieve(query, dense_vector=vec, filters=plan.filters, top_k=plan.top_k, source_types=st, mode=selected_mode)

    def _source_balanced_retrieve(self, query, plan, source_types):
        balanced = SourceBalancedRetriever(
            self.retriever,
            mode=self.retrieval_mode,
            per_source_k=min(plan.top_k, max(1, self.candidate_k)),
            final_k=plan.top_k,
            candidate_k=self.candidate_k,
        )
        return balanced.retrieve(query, dense_vector=self._embed(query), filters=plan.filters, source_types=source_types)

    def _should_use_source_balanced(self, plan, source_types) -> bool:
        return plan.retrieval_route in {"mixed", "multi_source_risk"} and len(source_types or []) > 1

    def _finalize_evidence(self, query, evidence, top_k):
        deduped = dedupe_chunks(evidence)
        reranker = getattr(self.retriever, "reranker", None) if self.retriever is not None else None
        if self.retrieval_mode == "hybrid_rerank" and reranker is not None:
            return deduped, dedupe_chunks(reranker.rerank(query, deduped, top_k))[:top_k], {"tool": "final_rerank", "payload": {"reranker_provider": self._reranker_provider(), "input_evidence_count": len(deduped), "output_evidence_count": min(len(deduped), top_k)}}
        return deduped, deduped[:top_k], None
    def run(self, query: str) -> AgentState:
        state=AgentState(query=query); state.retrieval_mode=self.retrieval_mode; state.candidate_k=self.candidate_k; state.reranker_provider=self._reranker_provider()
        cls=classify_query(query); state.normalized_query=cls.normalized_query; state.add_trace("normalize_query"); state.query_type=cls.query_type; state.expected_answer_type=cls.expected_answer_type; state.retrieval_route=cls.retrieval_route; state.add_trace("classify_query")
        plan=build_query_plan(query, cls, self.default_top_k); state.retrieval_plan.append(plan); state.add_trace("plan_retrieval")
        if plan.retrieval_route=="sql":
            state.sql_results=SQLRouter(self.duckdb_store).run(plan); state.add_tool_call("sql_router", plan.filters); state.add_trace("sql_lookup")
        elif plan.retrieval_route=="hybrid":
            if self.embedding_provider is not None: state.add_trace("query_embedding")
            state.retrieved_evidence=dedupe_chunks(self._retrieve(plan.query, plan)); state.add_tool_call("hybrid_retriever", self._payload(plan.top_k, plan.source_types)); state.add_trace("hybrid_retrieval")
        elif plan.retrieval_route=="mixed":
            if self.duckdb_store is not None:
                state.sql_results=SQLRouter(self.duckdb_store).run(plan); state.add_tool_call("sql_router", plan.filters); state.add_trace("sql_lookup")
            sources=plan.source_types or []
            if self._should_use_source_balanced(plan, sources):
                state.retrieved_evidence=dedupe_chunks(self._source_balanced_retrieve(plan.query, plan, sources)); state.add_tool_call("source_balanced_retriever", self._payload(plan.top_k, sources)); state.add_trace("mixed_source_balanced_retrieval"); state.add_trace("mixed_hybrid_retrieval")
            else:
                state.retrieved_evidence=dedupe_chunks(self._retrieve(plan.query, plan)); state.add_tool_call("hybrid_retriever", self._payload(plan.top_k, plan.source_types)); state.add_trace("mixed_hybrid_retrieval")
        elif plan.retrieval_route=="multi_source_risk":
            sources=plan.source_types or ["trademark","patent","litigation"]
            if self._should_use_source_balanced(plan, sources):
                state.retrieved_evidence=dedupe_chunks(self._source_balanced_retrieve(plan.query, plan, sources)); state.add_tool_call("source_balanced_retriever", self._payload(plan.top_k, sources)); state.add_trace("multi_source_risk_source_balanced_retrieval"); state.add_trace("multi_source_risk_retrieval")
            else:
                state.retrieved_evidence=dedupe_chunks(self._retrieve(plan.query, plan, source_types=sources)); state.add_tool_call("hybrid_retriever", self._payload(plan.top_k, sources)); state.add_trace("multi_source_risk_retrieval")
        ev=evaluate_evidence(plan, state.retrieved_evidence, state.sql_results); state.evidence_gaps=ev.evidence_gaps; state.add_trace("evaluate_evidence")
        while not ev.is_sufficient and state.iterations < self.max_iterations and self.retriever is not None:
            state.iterations += 1
            for st, fq in zip(ev.missing_source_types, ev.followup_queries):
                state.add_trace("query_rewrite")
                more=self._retrieve(fq, plan, source_types=[st]); state.retrieved_evidence=dedupe_chunks([*state.retrieved_evidence,*more]); state.add_tool_call("hybrid_retriever", self._payload(plan.top_k,[st],fq)); state.add_trace("followup_retrieval")
            ev=evaluate_evidence(plan, state.retrieved_evidence, state.sql_results); state.evidence_gaps=ev.evidence_gaps; state.add_trace("evaluate_evidence")
        state.retrieved_evidence, state.reranked_evidence, final_call = self._finalize_evidence(plan.query, state.retrieved_evidence, plan.top_k)
        if final_call is not None:
            state.add_tool_call(final_call["tool"], final_call["payload"]); state.add_trace("final_rerank")
        evidence=state.reranked_evidence or dedupe_chunks(state.retrieved_evidence)
        state.answer,state.citations=synthesize_answer(plan, state.sql_results, evidence, state.evidence_gaps); state.add_trace("final_answer")
        return state
