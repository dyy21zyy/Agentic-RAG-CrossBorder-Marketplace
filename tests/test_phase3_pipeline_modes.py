from __future__ import annotations

import importlib.util
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_common():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("agentic_rag_cli_common", ROOT / "scripts" / "agentic_rag_cli_common.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def args(mode: str, use_llm: bool = False) -> Namespace:
    return Namespace(
        pipeline_mode=mode,
        query="What trademark and patent risks should a seller consider for a smart travel bag product?",
        duckdb_path=None,
        chunks_path=None,
        use_milvus=False,
        collection_name="ip_chunks",
        embedding_provider="fake",
        retrieval_mode="bm25_only",
        candidate_k=50,
        top_k=3,
        max_iterations=2,
        reranker_provider="noop",
        reranker_model=None,
        source_types="trademark,patent,litigation",
        use_llm=use_llm,
        llm_provider="template",
        llm_model=None,
        demo=True,
    )


def test_run_agentic_rag_help_shows_pipeline_mode():
    cp = subprocess.run([sys.executable, "scripts/run_agentic_rag.py", "--help"], cwd=ROOT, text=True, capture_output=True)
    assert cp.returncode == 0
    assert "--pipeline-mode" in cp.stdout


def test_chat_agentic_rag_help_shows_pipeline_mode():
    cp = subprocess.run([sys.executable, "scripts/chat_agentic_rag.py", "--help"], cwd=ROOT, text=True, capture_output=True)
    assert cp.returncode == 0
    assert "--pipeline-mode" in cp.stdout


def test_agentic_mode_calls_agentic_rag_run(monkeypatch):
    common = load_common()
    called = {"run": 0}
    original = common.AgenticRAG.run

    def wrapped(self, query):
        called["run"] += 1
        return original(self, query)

    monkeypatch.setattr(common.AgenticRAG, "run", wrapped)
    result = common.run_pipeline("Explain trademark infringement", args("agentic"))
    assert called["run"] == 1
    assert result["pipeline_mode"] == "agentic"
    assert result["agent_enabled"] is True
    assert {"classify_query", "plan_retrieval", "evaluate_evidence"} <= set(result["trace"])


def test_basic_rag_does_not_call_agentic_run_and_calls_retriever(monkeypatch):
    common = load_common()
    calls = {"retrieve": 0}

    def forbidden_run(self, query):
        raise AssertionError("AgenticRAG.run must not be called in basic_rag mode")

    original_retrieve = common.HybridRetriever.retrieve

    def wrapped_retrieve(self, *a, **kw):
        calls["retrieve"] += 1
        return original_retrieve(self, *a, **kw)

    monkeypatch.setattr(common.AgenticRAG, "run", forbidden_run)
    monkeypatch.setattr(common.HybridRetriever, "retrieve", wrapped_retrieve)
    result = common.run_pipeline("smart travel bag risk", args("basic_rag"))
    assert calls["retrieve"] == 1
    assert result["pipeline_mode"] == "basic_rag"
    assert result["agent_enabled"] is False
    assert result["retrieval_route"] == "direct_retrieval"
    assert "basic_rag_direct_retrieval" in result["trace"]


def test_both_modes_use_same_grounded_answer_function(monkeypatch):
    common = load_common()
    seen = []

    def fake_grounded(query, deterministic_answer, evidence, provider, model):
        seen.append(query)
        return "shared llm answer", None

    monkeypatch.setattr(common, "generate_grounded_answer", fake_grounded)
    agentic = common.run_pipeline("Explain trademark infringement", args("agentic", use_llm=True))
    basic = common.run_pipeline("Explain trademark infringement", args("basic_rag", use_llm=True))
    assert seen == ["Explain trademark infringement", "Explain trademark infringement"]
    assert agentic["llm_answer"] == basic["llm_answer"] == "shared llm answer"


def test_policy_evidence_is_not_required_by_default():
    common = load_common()
    result = common.run_pipeline("smart travel bag risk", args("basic_rag"))
    payload = result["tool_calls"][0]["payload"]
    assert payload["source_types"] == ["trademark", "patent", "litigation"]
    assert ("p" + "olicy") not in payload["source_types"]
