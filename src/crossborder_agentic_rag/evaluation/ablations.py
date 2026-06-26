"""Ablation experiment runner."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Callable
from crossborder_agentic_rag.evaluation.datasets import EvalExample
from crossborder_agentic_rag.evaluation.evaluator import evaluate_agent

@dataclass(slots=True)
class AblationConfig:
    name: str
    retrieval_mode: str | None = None
    reranker_provider: str | None = None
    source_types: list[str] | None = None
    rrf_k: int | None = None
    disable_sql: bool = False
    disable_hybrid: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class AblationResult:
    name: str
    summary_metrics: dict[str, float]
    num_examples: int
    config: dict[str, Any]

def default_ablation_configs() -> list[AblationConfig]:
    all_src=["trademark","patent","policy","litigation"]
    return [
        AblationConfig("bm25_only", retrieval_mode="bm25", reranker_provider="none"),
        AblationConfig("dense_only", retrieval_mode="dense", reranker_provider="none"),
        AblationConfig("hybrid_rrf", retrieval_mode="hybrid_rrf", reranker_provider="none"),
        AblationConfig("hybrid_rerank", retrieval_mode="hybrid_rerank", reranker_provider="lexical"),
        AblationConfig("sql_only", disable_hybrid=True),
        AblationConfig("hybrid_only", disable_sql=True),
        AblationConfig("sql_plus_hybrid", retrieval_mode="hybrid_rrf"),
        AblationConfig("no_reranker", retrieval_mode="hybrid_rrf", reranker_provider="none"),
        AblationConfig("lexical_reranker", retrieval_mode="hybrid_rerank", reranker_provider="lexical"),
        AblationConfig("without_trademark", source_types=[s for s in all_src if s!="trademark"]),
        AblationConfig("without_patent", source_types=[s for s in all_src if s!="patent"]),
        AblationConfig("without_policy", source_types=[s for s in all_src if s!="policy"]),
        AblationConfig("without_litigation", source_types=[s for s in all_src if s!="litigation"]),
        AblationConfig("rrf_k_10", retrieval_mode="hybrid_rrf", rrf_k=10),
        AblationConfig("rrf_k_30", retrieval_mode="hybrid_rrf", rrf_k=30),
        AblationConfig("rrf_k_60", retrieval_mode="hybrid_rrf", rrf_k=60),
        AblationConfig("rrf_k_100", retrieval_mode="hybrid_rrf", rrf_k=100),
    ]

def run_ablations(examples: list[EvalExample], agent_factory: Callable[[AblationConfig], Any], configs: list[AblationConfig] | None = None, top_ks: list[int] | None = None) -> list[AblationResult]:
    out=[]
    for cfg in (configs or default_ablation_configs()):
        try:
            _, summary=evaluate_agent(agent_factory(cfg), examples, top_ks=top_ks)
            metrics=summary.metrics; config=asdict(cfg)
        except Exception as exc:
            metrics={"failed":1.0}; config=asdict(cfg); config.setdefault("metadata",{})["error"]=str(exc)
        out.append(AblationResult(cfg.name, metrics, len(examples), config))
    return out
