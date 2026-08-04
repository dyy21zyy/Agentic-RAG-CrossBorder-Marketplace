"""Config-driven construction for the single-turn risk-screening runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crossborder_agentic_rag.agentic.dispatcher import ToolDispatcher
from crossborder_agentic_rag.agentic.runtime import RiskScreeningRuntime
from crossborder_agentic_rag.config.registry import PluginRegistry
from crossborder_agentic_rag.config.settings import load_app_config
from crossborder_agentic_rag.llm.chat_client import TemplateChatClient, build_chat_client
from crossborder_agentic_rag.observability.jsonl_trace import LocalJsonlTraceSink
from crossborder_agentic_rag.observability.trace import TraceSink
from crossborder_agentic_rag.retrieval.bm25 import LocalBM25Retriever
from crossborder_agentic_rag.schemas import EvidenceChunk


class _BM25RetrieverAdapter:
    def __init__(self, retriever: LocalBM25Retriever, top_k: int = 8) -> None:
        self.retriever = retriever
        self.top_k = top_k

    def retrieve(self, query: str, source_types: list[str] | None = None) -> list[EvidenceChunk]:
        return self.retriever.search(query, source_types=source_types, top_k=self.top_k)


def _load_chunks_jsonl(path: str | Path) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(EvidenceChunk.from_dict(json.loads(line)))
    return chunks


def build_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register(
        "llm",
        "template",
        lambda cfg: TemplateChatClient(
            provider=str(cfg.get("provider") or "template"),
            model=str(cfg.get("model") or "template"),
        ),
    )
    registry.register(
        "llm",
        "openai_compatible",
        lambda cfg: build_chat_client(
            provider=str(cfg.get("provider") or "openai_compatible"),
            api_key=cfg.get("api_key"),
            base_url=cfg.get("base_url"),
            model=cfg.get("model"),
            disable_thinking=bool(cfg.get("disable_thinking", True)),
        ),
    )
    registry.register(
        "retrieval",
        "local_bm25_jsonl",
        lambda cfg: _BM25RetrieverAdapter(
            LocalBM25Retriever(_load_chunks_jsonl(cfg["chunks_path"])),
            top_k=int(cfg.get("top_k") or 8),
        ),
    )
    return registry


def build_offline_template_runtime(trace_sink: TraceSink | None = None) -> RiskScreeningRuntime:
    return RiskScreeningRuntime(
        dispatcher=ToolDispatcher(),
        llm=TemplateChatClient(),
        trace_sink=trace_sink,
    )


def build_runtime_from_config(config_path: str | Path) -> RiskScreeningRuntime:
    cfg = load_app_config(config_path)
    registry = build_registry()
    llm_cfg: dict[str, Any] = {"provider": "template", **cfg.llm}
    llm = registry.build("llm", llm_cfg)
    retrieval_cfg = dict(cfg.retrieval)
    if not retrieval_cfg.get("provider") and not retrieval_cfg.get("name"):
        raise ValueError(
            "retrieval provider is not configured; use --offline-template for explicit offline fallback"
        )
    retriever = registry.build("retrieval", retrieval_cfg)
    dispatcher = ToolDispatcher(retriever=retriever)
    trace_sink = None
    if cfg.observability.get("provider") == "local_jsonl":
        trace_path = Path(str(cfg.observability.get("path") or "traces/local.jsonl"))
        trace_sink = LocalJsonlTraceSink(trace_path)
    return RiskScreeningRuntime(dispatcher=dispatcher, llm=llm, trace_sink=trace_sink)
