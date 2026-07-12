#!/usr/bin/env python
from __future__ import annotations


def _ragas_context_from_evidence_v2(e, max_chars: int = 1200) -> str:
    """Convert one evidence item into one RAGAS context string."""
    if e is None:
        return ""

    if isinstance(e, dict):
        d = e
    elif hasattr(e, "model_dump"):
        try:
            d = e.model_dump()
        except Exception:
            d = {}
    elif hasattr(e, "dict"):
        try:
            d = e.dict()
        except Exception:
            d = {}
    else:
        d = {
            "chunk_id": getattr(e, "chunk_id", None) or getattr(e, "id", None),
            "id": getattr(e, "id", None),
            "title": getattr(e, "title", None),
            "content": getattr(e, "content", None) or getattr(e, "text", None) or getattr(e, "page_content", None),
            "source_type": getattr(e, "source_type", None),
            "source_subtype": getattr(e, "source_subtype", None),
            "metadata": getattr(e, "metadata", None) or {},
        }

    md = d.get("metadata") or {}
    if not isinstance(md, dict):
        md = {}

    cid = d.get("chunk_id") or d.get("id") or md.get("chunk_id") or ""
    title = d.get("title") or md.get("title") or ""
    content = (
        d.get("content")
        or d.get("text")
        or d.get("page_content")
        or d.get("snippet")
        or md.get("content")
        or ""
    )
    source_type = d.get("source_type") or md.get("source_type") or ""
    source_subtype = d.get("source_subtype") or md.get("source_subtype") or ""

    parts = []
    if cid:
        parts.append(f"chunk_id: {cid}")
    if source_type or source_subtype:
        parts.append(f"source: {source_type}/{source_subtype}")
    if title:
        parts.append(f"title: {title}")
    if content:
        parts.append(str(content))

    return "\n".join(parts).strip()[:max_chars]


def _ragas_reference_from_scope_v2(local_vars) -> str:
    """Build RAGAS reference from the original eval row if available."""
    row = {}

    for obj in local_vars.values():
        if isinstance(obj, dict) and (
            "gold_answer" in obj
            or "gold_answer_key_points" in obj
            or "answer_key_points" in obj
            or "relevant_chunk_ids" in obj
            or "target_entities" in obj
        ):
            row = obj
            break

    parts = []

    if row.get("gold_answer"):
        parts.append(str(row.get("gold_answer")))

    key_points = row.get("gold_answer_key_points") or row.get("answer_key_points") or []
    if key_points:
        parts.append("Key points: " + "; ".join(map(str, key_points)))

    target_entities = row.get("target_entities") or {}
    if target_entities:
        parts.append("Target entities: " + str(target_entities))

    relevant = row.get("relevant_chunk_ids") or []
    if relevant:
        parts.append("Relevant chunk ids: " + "; ".join(map(str, relevant[:20])))

    strict = row.get("strict_relevant_chunk_ids") or []
    if strict:
        parts.append("Strict relevant chunk ids: " + "; ".join(map(str, strict[:20])))

    return "\n".join(parts).strip()


import argparse
import csv
import gc
import json
import math
import re
import statistics
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag_cli_common import build_runtime, load_env
from crossborder_agentic_rag.llm.chat_client import build_chat_client


QUERY_KEYS = ["query", "question", "user_query", "input", "prompt"]
GOLD_ANSWER_KEYS = ["gold_answer", "expected_answer", "reference_answer", "answer"]
GOLD_CHUNK_KEYS = [
    "gold_chunk_ids",
    "expected_chunk_ids",
    "relevant_chunk_ids",
    "positive_chunk_ids",
    "chunk_ids",
]
QRELS_KEYS = ["qrels", "relevance", "judgments", "labels"]
SOURCE_KEYS = ["required_source_types", "expected_source_types", "source_types"]
KS = [1, 3, 5, 10]


# -----------------------------
# Timing wrappers
# -----------------------------
class TimingMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.data = {
            "retrieval_ms": 0.0,
            "retrieval_calls": 0,
            "rerank_ms": 0.0,
            "rerank_calls": 0,
            "sql_ms": 0.0,
            "sql_calls": 0,
        }

    def add(self, key_ms: str, key_calls: str, seconds: float) -> None:
        self.data[key_ms] += seconds * 1000
        self.data[key_calls] += 1


class TimedRetriever:
    def __init__(self, inner: Any, meter: TimingMeter) -> None:
        self.inner = inner
        self.meter = meter

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return self.inner.retrieve(*args, **kwargs)
        finally:
            self.meter.add("retrieval_ms", "retrieval_calls", time.perf_counter() - t0)


class TimedReranker:
    def __init__(self, inner: Any, meter: TimingMeter) -> None:
        self.inner = inner
        self.meter = meter

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def rerank(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return self.inner.rerank(*args, **kwargs)
        finally:
            self.meter.add("rerank_ms", "rerank_calls", time.perf_counter() - t0)


class TimedDuckDBStore:
    def __init__(self, inner: Any, meter: TimingMeter) -> None:
        self.inner = inner
        self.meter = meter

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.inner, name)
        if not callable(attr):
            return attr

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return attr(*args, **kwargs)
            finally:
                self.meter.add("sql_ms", "sql_calls", time.perf_counter() - t0)

        return wrapped


def attach_timing(runtime: Any, meter: TimingMeter) -> None:
    if runtime.duckdb_store is not None:
        timed_duck = TimedDuckDBStore(runtime.duckdb_store, meter)
        runtime.duckdb_store = timed_duck
        if runtime.agent is not None and hasattr(runtime.agent, "duckdb_store"):
            runtime.agent.duckdb_store = timed_duck

    if runtime.retriever is not None:
        reranker = getattr(runtime.retriever, "reranker", None)
        if reranker is not None:
            runtime.retriever.reranker = TimedReranker(reranker, meter)

        timed_retriever = TimedRetriever(runtime.retriever, meter)
        runtime.retriever = timed_retriever
        if runtime.agent is not None and hasattr(runtime.agent, "retriever"):
            runtime.agent.retriever = timed_retriever


# -----------------------------
# Data loading
# -----------------------------
def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def get_first(row: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def extract_query(row: dict[str, Any]) -> str:
    q = get_first(row, QUERY_KEYS)
    if q is None:
        raise ValueError(f"Cannot find query field. Row keys={list(row.keys())}")
    return str(q)


def extract_gold_answer(row: dict[str, Any]) -> str | None:
    value = get_first(row, GOLD_ANSWER_KEYS)
    return str(value) if value else None


def extract_required_sources(row: dict[str, Any]) -> set[str]:
    value = get_first(row, SOURCE_KEYS)
    if value is None:
        return set()
    if isinstance(value, str):
        return {x.strip() for x in value.split(",") if x.strip()}
    if isinstance(value, list):
        return {str(x).strip() for x in value if str(x).strip()}
    return set()


def extract_qrels(row: dict[str, Any]) -> dict[str, float]:
    qrels: dict[str, float] = {}

    value = get_first(row, QRELS_KEYS)
    if value is not None:
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (int, float)):
                    qrels[str(k)] = float(v)
                elif isinstance(v, dict):
                    cid = v.get("chunk_id") or v.get("doc_id") or k
                    score = v.get("score") or v.get("label") or v.get("relevance") or 1
                    try:
                        qrels[str(cid)] = float(score)
                    except Exception:
                        qrels[str(cid)] = 1.0
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    qrels[item] = 1.0
                elif isinstance(item, dict):
                    cid = item.get("chunk_id") or item.get("doc_id") or item.get("id")
                    if cid:
                        score = item.get("score") or item.get("label") or item.get("relevance") or 1
                        try:
                            qrels[str(cid)] = float(score)
                        except Exception:
                            qrels[str(cid)] = 1.0

    gold_ids_value = get_first(row, GOLD_CHUNK_KEYS)
    if gold_ids_value is not None:
        if isinstance(gold_ids_value, str):
            qrels.setdefault(gold_ids_value, 1.0)
        elif isinstance(gold_ids_value, list):
            for item in gold_ids_value:
                if isinstance(item, str):
                    qrels.setdefault(item, 1.0)
                elif isinstance(item, dict):
                    cid = item.get("chunk_id") or item.get("doc_id") or item.get("id")
                    if cid:
                        qrels.setdefault(str(cid), 1.0)

    return {k: v for k, v in qrels.items() if v > 0}


# -----------------------------
# Retrieval metrics
# -----------------------------
def ids_from_evidence(items: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for x in items or []:
        chunk_id = x.get("chunk_id")
        doc_id = x.get("doc_id")
        if chunk_id:
            ids.append(str(chunk_id))
        elif doc_id:
            ids.append(str(doc_id))
    return ids


def expanded_ids_from_evidence(items: list[dict[str, Any]]) -> list[set[str]]:
    out: list[set[str]] = []
    for x in items or []:
        s = set()
        if x.get("chunk_id"):
            s.add(str(x["chunk_id"]))
        if x.get("doc_id"):
            s.add(str(x["doc_id"]))
        out.append(s)
    return out


def sources_from_evidence(items: list[dict[str, Any]]) -> set[str]:
    return {str(x.get("source_type")) for x in items or [] if x.get("source_type")}


def precision_at_k(ranked: list[set[str]], qrels: dict[str, float], k: int) -> float | None:
    if not qrels:
        return None
    top = ranked[:k]
    hits = sum(1 for ids in top if ids & qrels.keys())
    return hits / k


def recall_at_k(ranked: list[set[str]], qrels: dict[str, float], k: int) -> float | None:
    if not qrels:
        return None
    hit_ids = set()
    for ids in ranked[:k]:
        hit_ids |= ids & qrels.keys()
    return len(hit_ids) / len(qrels)


def hit_at_k(ranked: list[set[str]], qrels: dict[str, float], k: int) -> float | None:
    if not qrels:
        return None
    return 1.0 if any(ids & qrels.keys() for ids in ranked[:k]) else 0.0


def mrr_at_k(ranked: list[set[str]], qrels: dict[str, float], k: int) -> float | None:
    if not qrels:
        return None
    for i, ids in enumerate(ranked[:k], start=1):
        if ids & qrels.keys():
            return 1.0 / i
    return 0.0


def dcg_at_k(gains: list[float], k: int) -> float:
    total = 0.0
    for i, g in enumerate(gains[:k], start=1):
        total += g / math.log2(i + 1)
    return total


def ndcg_at_k(ranked: list[set[str]], qrels: dict[str, float], k: int) -> float | None:
    if not qrels:
        return None

    gains = []
    for ids in ranked[:k]:
        gains.append(max([qrels[x] for x in ids if x in qrels], default=0.0))

    ideal = sorted(qrels.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(gains, k) / ideal_dcg


def map_at_k(ranked: list[set[str]], qrels: dict[str, float], k: int) -> float | None:
    if not qrels:
        return None

    hits = 0
    precisions = []
    seen_relevant = set()

    for i, ids in enumerate(ranked[:k], start=1):
        matched = ids & qrels.keys()
        new_matched = matched - seen_relevant
        if new_matched:
            hits += 1
            seen_relevant |= new_matched
            precisions.append(hits / i)

    if not precisions:
        return 0.0

    denom = min(len(qrels), k)
    return sum(precisions) / denom


def compute_retrieval_metrics(items: list[dict[str, Any]], qrels: dict[str, float]) -> dict[str, Any]:
    ranked = expanded_ids_from_evidence(items)
    out: dict[str, Any] = {"gold_relevant_count": len(qrels)}

    for k in KS:
        out[f"precision_at_{k}"] = precision_at_k(ranked, qrels, k)
        out[f"recall_at_{k}"] = recall_at_k(ranked, qrels, k)
        out[f"hit_at_{k}"] = hit_at_k(ranked, qrels, k)
        out[f"mrr_at_{k}"] = mrr_at_k(ranked, qrels, k)
        out[f"ndcg_at_{k}"] = ndcg_at_k(ranked, qrels, k)
        out[f"map_at_{k}"] = map_at_k(ranked, qrels, k)

    return out


# -----------------------------
# Generation metrics
# -----------------------------
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.lower() for x in _WORD_RE.findall(text)]


def token_f1(pred: str | None, gold: str | None) -> float | None:
    p = tokenize(pred)
    g = tokenize(gold)
    if not p or not g:
        return None

    g_count: dict[str, int] = {}
    for tok in g:
        g_count[tok] = g_count.get(tok, 0) + 1

    common = 0
    for tok in p:
        if g_count.get(tok, 0) > 0:
            common += 1
            g_count[tok] -= 1

    if common == 0:
        return 0.0

    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def jaccard(a: str | None, b: str | None) -> float | None:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return None
    return len(ta & tb) / len(ta | tb)


def evidence_text(items: list[dict[str, Any]], max_items: int = 8, max_chars_each: int = 500) -> str:
    lines = []
    for i, x in enumerate(items[:max_items], start=1):
        title = x.get("title") or ""
        content = x.get("content_preview") or x.get("content") or ""
        source = x.get("source_type") or ""
        cid = x.get("chunk_id") or x.get("doc_id") or ""
        lines.append(f"[{i}] {source} {cid} {title}: {str(content)[:max_chars_each]}")
    return "\n".join(lines)


def sentence_split(text: str | None) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if len(p.strip()) >= 8]


def heuristic_faithfulness(answer: str | None, items: list[dict[str, Any]]) -> float | None:
    if not answer or not items:
        return None

    ev_text = evidence_text(items, max_items=20, max_chars_each=1000)
    ev_tokens = set(tokenize(ev_text))
    claims = sentence_split(answer)

    if not claims:
        return None

    supported = 0
    for claim in claims:
        ctoks = set(tokenize(claim))
        if not ctoks:
            continue
        overlap = len(ctoks & ev_tokens) / max(1, len(ctoks))
        if overlap >= 0.35:
            supported += 1

    return supported / len(claims)


def heuristic_context_relevance(query: str, items: list[dict[str, Any]]) -> float | None:
    if not query or not items:
        return None

    scores = []
    for x in items:
        text = f"{x.get('title','')} {x.get('content_preview','')} {x.get('content','')}"
        score = jaccard(query, text)
        if score is not None:
            scores.append(score)

    if not scores:
        return None

    return statistics.mean(scores[:5])


def heuristic_answer_relevance(query: str, answer: str | None) -> float | None:
    return jaccard(query, answer)


def parse_llm_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
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

    raise ValueError("Cannot parse LLM judge JSON")


def call_llm_text(client: Any, prompt: str) -> str:
    if hasattr(client, "invoke"):
        resp = client.invoke(prompt)
    elif hasattr(client, "complete"):
        resp = client.complete(prompt)
    elif hasattr(client, "generate"):
        resp = client.generate(prompt)
    elif callable(client):
        resp = client(prompt)
    else:
        raise TypeError("Unsupported LLM client")

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


def llm_judge_generation(
    client: Any,
    query: str,
    answer: str | None,
    context: str,
    gold_answer: str | None = None,
) -> dict[str, Any]:
    if not answer:
        return {}

    prompt = f"""You are an evaluator for a RAG system.

Evaluate the answer using the retrieved context.

Return JSON only:
{{
  "faithfulness": number between 0 and 1,
  "answer_relevance": number between 0 and 1,
  "context_relevance": number between 0 and 1,
  "document_relevance": number between 0 and 1,
  "completeness": number between 0 and 1,
  "reason": "brief explanation"
}}

Definitions:
- faithfulness: whether the answer is supported by the retrieved context and does not invent facts.
- answer_relevance: whether the answer directly addresses the user query.
- context_relevance/document_relevance: whether the retrieved documents are relevant to the query.
- completeness: whether the answer covers the important aspects needed by the query.

User query:
{query}

Retrieved context:
{context}

Answer:
{answer}

Reference answer, if available:
{gold_answer or "N/A"}
"""
    raw = call_llm_text(client, prompt)
    data = parse_llm_json(raw)

    out = {}
    for k in ["faithfulness", "answer_relevance", "context_relevance", "document_relevance", "completeness"]:
        try:
            out[f"llm_{k}"] = float(data.get(k))
        except Exception:
            out[f"llm_{k}"] = None

    out["llm_judge_reason"] = str(data.get("reason") or "")
    return out


def compute_generation_metrics(
    query: str,
    answer: str | None,
    final_items: list[dict[str, Any]],
    gold_answer: str | None,
    judge_client: Any | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "answer_token_f1": token_f1(answer, gold_answer),
        "heuristic_faithfulness": heuristic_faithfulness(answer, final_items),
        "heuristic_answer_relevance": heuristic_answer_relevance(query, answer),
        "heuristic_context_relevance": heuristic_context_relevance(query, final_items),
    }

    if judge_client is not None:
        try:
            out.update(
                llm_judge_generation(
                    judge_client,
                    query=query,
                    answer=answer,
                    context=evidence_text(final_items),
                    gold_answer=gold_answer,
                )
            )
        except Exception as exc:
            out["llm_judge_error"] = str(exc)

    return out


# -----------------------------
# Runtime
# -----------------------------
def make_runtime_args(base: argparse.Namespace, mode: str) -> Namespace:
    return Namespace(
        pipeline_mode=mode,
        query="placeholder",
        duckdb_path=base.duckdb_path,
        chunks_path=base.chunks_path,
        use_milvus=base.use_milvus,
        collection_name=base.collection_name,
        embedding_provider=base.embedding_provider,
        retrieval_mode=base.retrieval_mode,
        candidate_k=base.candidate_k,
        top_k=base.top_k,
        max_iterations=base.max_iterations,
        reranker_provider=base.reranker_provider,
        reranker_model=base.reranker_model,
        source_types=base.source_types,
        use_llm=base.use_llm,
        llm_provider=base.llm_provider,
        llm_model=base.llm_model,
        llm_base_url=base.llm_base_url,
        max_evidence_for_llm=base.max_evidence_for_llm,
        max_chars_per_evidence=base.max_chars_per_evidence,
        llm_max_tokens=base.llm_max_tokens,
        temperature=base.temperature,
        demo=base.demo,
    )


def build_judge_client(args: argparse.Namespace) -> Any | None:
    if not args.judge_with_llm:
        return None

    return build_chat_client(
        provider=args.judge_provider or args.llm_provider,
        base_url=args.judge_base_url or args.llm_base_url,
        model=args.judge_model or args.llm_model,
        default_temperature=0.0,
        default_max_tokens=args.judge_max_tokens,
    )


def run_mode(
    mode: str,
    rows: list[dict[str, Any]],
    base_args: argparse.Namespace,
    judge_client: Any | None = None,
) -> list[dict[str, Any]]:
    args = make_runtime_args(base_args, mode)
    meter = TimingMeter()

    print(f"\n[build_runtime] mode={mode}", flush=True)
    runtime = build_runtime(args)
    attach_timing(runtime, meter)

    results: list[dict[str, Any]] = []

    for i, row in enumerate(rows, start=1):
        query = extract_query(row)
        qrels = extract_qrels(row)
        gold_answer = extract_gold_answer(row)
        required_sources = extract_required_sources(row)

        meter.reset()

        t0 = time.perf_counter()
        result = runtime.run_query(query)
        external_ms = (time.perf_counter() - t0) * 1000

        final_items = result.get("reranked_evidence") or result.get("retrieved_evidence") or []
        retrieved_items = result.get("retrieved_evidence") or []
        final_sources = sources_from_evidence(final_items)

        retrieval_metrics_final = compute_retrieval_metrics(final_items, qrels)
        retrieval_metrics_raw = {
            f"raw_{k}": v for k, v in compute_retrieval_metrics(retrieved_items, qrels).items()
        }

        source_coverage = None
        if required_sources:
            source_coverage = len(required_sources & final_sources) / len(required_sources)

        answer_text = result.get("llm_answer") or result.get("deterministic_answer")
        generation_metrics = compute_generation_metrics(
            query=query,
            answer=answer_text,
            final_items=final_items,
            gold_answer=gold_answer,
            judge_client=judge_client,
        )

        row_out: dict[str, Any] = {
            "idx": i,
            "mode": mode,
            "query": query,
            "query_type": result.get("query_type"),
            "expected_answer_type": result.get("expected_answer_type"),
            "retrieval_route": result.get("retrieval_route"),
            "retrieval_mode": result.get("retrieval_mode"),
            "latency_ms": result.get("latency_ms"),
            "external_latency_ms": round(external_ms, 2),
            "retrieval_ms": round(meter.data["retrieval_ms"], 2),
            "retrieval_calls": meter.data["retrieval_calls"],
            "rerank_ms": round(meter.data["rerank_ms"], 2),
            "rerank_calls": meter.data["rerank_calls"],
            "sql_ms": round(meter.data["sql_ms"], 2),
            "sql_calls": meter.data["sql_calls"],
            "other_ms": round(
                external_ms
                - meter.data["retrieval_ms"]
                - meter.data["rerank_ms"]
                - meter.data["sql_ms"],
                2,
            ),
            "tool_call_count": result.get("tool_call_count"),
            "followup_query_count": result.get("followup_query_count"),
            "retrieved_evidence_count": result.get("retrieved_evidence_count"),
            "final_evidence_count": result.get("final_evidence_count"),
            "evidence_gap_count": len(result.get("evidence_gaps") or []),
            "evidence_gaps": json.dumps(result.get("evidence_gaps") or [], ensure_ascii=False),
            "final_source_types": ",".join(sorted(final_sources)),
            "required_source_types": ",".join(sorted(required_sources)),
            "source_coverage": source_coverage,
            "gold_answer_available": bool(gold_answer),
            "qrels_available": bool(qrels),
            "trace": " -> ".join(result.get("trace") or []),
            "tools": ",".join([x.get("tool", "") for x in result.get("tool_calls") or []]),
            "answer_preview": str(answer_text or "")[:500].replace("\n", " "),
        }

        row_out.update(retrieval_metrics_final)
        row_out.update(retrieval_metrics_raw)
        row_out.update(generation_metrics)

        # RAGAS export fields
        ragas_contexts = [
            _ragas_context_from_evidence_v2(e)
            for e in (final_items or [])[:5]
        ]
        ragas_contexts = [c for c in ragas_contexts if c]

        row_out["ragas_user_input"] = row_out.get("query", "")
        row_out["ragas_response"] = str(answer_text or "")
        row_out["ragas_retrieved_contexts"] = ragas_contexts
        row_out["ragas_reference"] = _ragas_reference_from_scope_v2(locals())
        row_out["ragas_context_count"] = len(ragas_contexts)
        row_out["ragas_empty_response"] = not bool(row_out["ragas_response"])
        row_out["ragas_empty_contexts"] = not bool(ragas_contexts)

        # Explicit LLM/deterministic provenance fields
        row_out["llm_provider_runtime"] = result.get("llm_provider")
        row_out["llm_model_runtime"] = result.get("llm_model")
        row_out["llm_answer_present"] = bool(result.get("llm_answer"))
        row_out["deterministic_answer_present"] = bool(
            result.get("deterministic_answer")
        )
        row_out["llm_error_runtime"] = result.get("llm_error")

        if result.get("llm_answer"):
            row_out["answer_source"] = "llm_answer"
        elif result.get("deterministic_answer"):
            row_out["answer_source"] = "deterministic_answer"
        else:
            row_out["answer_source"] = "unknown"

        results.append(row_out)

        print(
            f"[{mode}] {i}/{len(rows)} "
            f"lat={row_out['external_latency_ms']}ms "
            f"ret={row_out['retrieval_ms']}ms "
            f"rerank={row_out['rerank_ms']}ms "
            f"sql={row_out['sql_ms']}ms "
            f"P@5={row_out.get('precision_at_5')} "
            f"R@5={row_out.get('recall_at_5')} "
            f"MRR@5={row_out.get('mrr_at_5')} "
            f"faith={row_out.get('llm_faithfulness', row_out.get('heuristic_faithfulness'))} "
            f"gaps={row_out['evidence_gap_count']} "
            f"tools={row_out['tool_call_count']}",
            flush=True,
        )

    del runtime
    gc.collect()
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    for mode in sorted({r["mode"] for r in results}):
        group = [r for r in results if r["mode"] == mode]

        def nums(key: str) -> list[float]:
            values = []
            for r in group:
                v = r.get(key)
                if v is None or v == "":
                    continue
                try:
                    values.append(float(v))
                except Exception:
                    pass
            return values

        mode_summary: dict[str, Any] = {"n": len(group)}

        metric_keys = [
            "external_latency_ms",
            "latency_ms",
            "retrieval_ms",
            "rerank_ms",
            "sql_ms",
            "other_ms",
            "tool_call_count",
            "followup_query_count",
            "retrieved_evidence_count",
            "final_evidence_count",
            "evidence_gap_count",
            "source_coverage",
            "answer_token_f1",
            "heuristic_faithfulness",
            "heuristic_answer_relevance",
            "heuristic_context_relevance",
            "llm_faithfulness",
            "llm_answer_relevance",
            "llm_context_relevance",
            "llm_document_relevance",
            "llm_completeness",
        ]

        for k in KS:
            metric_keys += [
                f"precision_at_{k}",
                f"recall_at_{k}",
                f"hit_at_{k}",
                f"mrr_at_{k}",
                f"ndcg_at_{k}",
                f"map_at_{k}",
            ]

        for key in metric_keys:
            values = nums(key)
            if values:
                mode_summary[f"{key}_mean"] = round(statistics.mean(values), 4)
                mode_summary[f"{key}_median"] = round(statistics.median(values), 4)

        summary[mode] = mode_summary

    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--modes", default="rule_based,agentic_llm")
    p.add_argument("--limit", type=int)

    p.add_argument("--chunks-path")
    p.add_argument("--duckdb-path")
    p.add_argument(
        "--retrieval-mode",
        choices=["bm25_only", "dense_only", "hybrid_rrf", "hybrid_rerank"],
        default="bm25_only",
    )
    p.add_argument("--use-milvus", action="store_true")
    p.add_argument("--collection-name", default="ip_chunks_qa_300k")
    p.add_argument("--embedding-provider", default="fake")
    p.add_argument("--reranker-provider", default="lexical")
    p.add_argument("--reranker-model")
    p.add_argument("--candidate-k", type=int, default=50)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-iterations", type=int, default=2)
    p.add_argument("--source-types", default="trademark,patent,litigation")

    p.add_argument("--use-llm", action="store_true")
    p.add_argument("--llm-provider", default="template")
    p.add_argument("--llm-model")
    p.add_argument("--llm-base-url")
    p.add_argument("--max-evidence-for-llm", type=int, default=6)
    p.add_argument("--max-chars-per-evidence", type=int, default=450)
    p.add_argument("--llm-max-tokens", type=int, default=800)
    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument("--judge-with-llm", action="store_true")
    p.add_argument("--judge-provider")
    p.add_argument("--judge-model")
    p.add_argument("--judge-base-url")
    p.add_argument("--judge-max-tokens", type=int, default=700)

    p.add_argument("--demo", action="store_true")

    args = p.parse_args()
    load_env()

    rows = load_jsonl(Path(args.queries), limit=args.limit)
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    judge_client = build_judge_client(args)

    all_results: list[dict[str, Any]] = []
    for mode in modes:
        all_results.extend(run_mode(mode, rows, args, judge_client=judge_client))

    csv_path = out_dir / "comparison_metrics.csv"
    jsonl_path = out_dir / "comparison_outputs.jsonl"
    summary_path = out_dir / "summary.json"

    if all_results:
        fieldnames = list(all_results[0].keys())

        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summary = summarize(all_results)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\nSaved:")
        print(csv_path)
        print(jsonl_path)
        print(summary_path)
        print("\nSummary:")
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
