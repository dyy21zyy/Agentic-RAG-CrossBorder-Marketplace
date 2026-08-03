# IP Risk Agentic RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the v1 "基于 Agentic RAG 的跨境电商知识产权风险初筛系统" described in `DEV_SPEC.md`, with structured risk reports, single-agent tool planning, MCP access, Streamlit inspection, Langfuse/local tracing, and evaluation gates.

**Architecture:** Keep the existing `crossborder_agentic_rag` package as the domain package. Add focused schema, report, agentic runtime, observability, MCP, dashboard service, and evaluation modules that communicate through stable dataclasses and JSON-compatible dictionaries. Reuse existing ingestion, retrieval, graph, DuckDB, Milvus, RRF, reranker, and RAGAS utilities where they already have tests.

**Tech Stack:** Python 3.10+, dataclasses, pytest, DuckDB, BM25, Milvus optional backend, NetworkX GraphRAG, OpenAI-compatible LLM clients, RAGAS optional evaluation, Langfuse optional tracing, Streamlit optional dashboard, MCP Python SDK optional server.

## Global Constraints

- The system name is `基于 Agentic RAG 的跨境电商知识产权风险初筛系统`.
- v1 is single-turn and single-agent tool-planning; no multi-turn memory and no multi-agent orchestration.
- Core evidence domains are `trademark`, `patent`, and `litigation`.
- Current source formats are XML, TSV, and CSV.
- Image support in v1 is a schema and enrichment contract; current text sources emit `images=[]`.
- Risk output is a structured `RiskScreeningReport`, not free-form QA.
- Risk verdicts are `no_risk_found`, `caution`, `not_recommended`, and `insufficient_evidence`.
- Reports are preliminary screening reports and must not claim final legal conclusions.
- LLM providers must support thinking-disabled behavior for Qwen/OpenAI-compatible usage when the backend accepts that option.
- MCP serves AI clients first; v1 does not add a production HTTP API.
- Langfuse is optional; local JSONL trace fallback is mandatory.
- RAGAS evaluates generation quality and does not replace retrieval or citation metrics.
- All runtime artifacts under `data/`, `reports/`, `traces/`, and generated eval outputs are local artifacts, not source package inputs.

---

## File Structure Map

Create or modify these files across the plan:

```text
src/crossborder_agentic_rag/schemas/images.py
src/crossborder_agentic_rag/schemas/evidence.py
src/crossborder_agentic_rag/schemas/documents.py
src/crossborder_agentic_rag/schemas/reports.py
src/crossborder_agentic_rag/schemas/traces.py
src/crossborder_agentic_rag/schemas/evaluation.py
src/crossborder_agentic_rag/schemas/__init__.py
src/crossborder_agentic_rag/config/settings.py
src/crossborder_agentic_rag/config/registry.py
src/crossborder_agentic_rag/llm/chat_client.py
src/crossborder_agentic_rag/agentic/normalizer.py
src/crossborder_agentic_rag/agentic/rewriter.py
src/crossborder_agentic_rag/agentic/planner.py
src/crossborder_agentic_rag/agentic/dispatcher.py
src/crossborder_agentic_rag/agentic/evidence_gap.py
src/crossborder_agentic_rag/agentic/runtime.py
src/crossborder_agentic_rag/reports/builder.py
src/crossborder_agentic_rag/reports/citation_audit.py
src/crossborder_agentic_rag/observability/trace.py
src/crossborder_agentic_rag/observability/jsonl_trace.py
src/crossborder_agentic_rag/observability/langfuse_trace.py
src/crossborder_agentic_rag/mcp_server/tools.py
src/crossborder_agentic_rag/mcp_server/resources.py
src/crossborder_agentic_rag/mcp_server/server.py
src/crossborder_agentic_rag/dashboard/services.py
src/crossborder_agentic_rag/dashboard/app.py
src/crossborder_agentic_rag/evaluation/citation_metrics.py
src/crossborder_agentic_rag/evaluation/agent_metrics.py
src/crossborder_agentic_rag/evaluation/eval_runner.py
scripts/query.py
scripts/evaluate.py
scripts/run_mcp_server.py
scripts/run_dashboard.py
configs/app.yaml
configs/plugins.yaml
tests/test_v1_schema_contracts.py
tests/test_v1_config_registry.py
tests/test_v1_llm_client.py
tests/test_v1_report_builder.py
tests/test_v1_agentic_runtime.py
tests/test_v1_trace_adapters.py
tests/test_v1_mcp_tools.py
tests/test_v1_dashboard_services.py
tests/test_v1_evaluation_runner.py
tests/test_v1_cli_contracts.py
```

Use the existing modules as adapters instead of deleting them during the v1 build. Each task below adds a testable capability without requiring all later tasks to exist.

---

### Task 1: Core Schema Contracts

**Files:**
- Create: `src/crossborder_agentic_rag/schemas/images.py`
- Create: `src/crossborder_agentic_rag/schemas/reports.py`
- Create: `src/crossborder_agentic_rag/schemas/traces.py`
- Create: `src/crossborder_agentic_rag/schemas/evaluation.py`
- Modify: `src/crossborder_agentic_rag/schemas/documents.py`
- Modify: `src/crossborder_agentic_rag/schemas/evidence.py`
- Modify: `src/crossborder_agentic_rag/schemas/__init__.py`
- Test: `tests/test_v1_schema_contracts.py`

**Interfaces:**
- Produces: `ImageAsset`, `EvidenceHit`, `RiskScreeningReport`, `RiskVerdict`, `TraceEvent`, `EvaluationRun`.
- Produces: `NormalizedDocument.images: list[ImageAsset]` and `EvidenceChunk.images: list[ImageAsset]`.
- Consumed by later tasks: report builder, MCP tools, dashboard services, trace adapters, evaluation runner.

**TDD Rhythm:** Do Step 1 first and change only the test file. Do not create schema implementation files until Step 2 has failed for the expected reason. Step 6 must rerun the same task test file and turn it green before the commit.

- [ ] **Step 1: Write the failing test**

```python
from crossborder_agentic_rag.schemas import (
    EvidenceChunk,
    EvidenceHit,
    ImageAsset,
    NormalizedDocument,
    RiskScreeningReport,
    RiskVerdict,
    TraceEvent,
)


def test_text_document_defaults_to_empty_images():
    doc = NormalizedDocument(
        doc_id="trademark:1",
        source_type="trademark",
        title="MARK",
        content="Goods and services evidence",
    )
    assert doc.images == []
    assert doc.to_dict()["images"] == []


def test_risk_report_required_fields_round_trip():
    hit = EvidenceHit(
        evidence_id="E1",
        chunk_id="trademark:1:chunk:0",
        source_type="trademark",
        title="Trademark evidence",
        content="Registered mark evidence",
        citation="[trademark:1:chunk:0] Trademark evidence",
        rank=1,
        score=1.0,
        retrieval_mode="bm25_only",
        tool_name="trademark_search_tool",
    )
    report = RiskScreeningReport(
        report_id="report-1",
        trace_id="trace-1",
        created_at="2026-08-03T00:00:00Z",
        product_profile={"name": "smart phone case"},
        target_markets=["US"],
        screening_scope=["trademark"],
        overall_verdict=RiskVerdict.CAUTION,
        country_summaries=[{"country": "US", "verdict": "caution", "summary": "Trademark evidence requires review."}],
        risk_cards={"no_risk_found": 0, "caution": 1, "not_recommended": 0, "insufficient_evidence": 0},
        module_results=[{"module": "trademark", "verdict": "caution", "evidence_ids": ["E1"]}],
        evidence_items=[hit],
        action_recommendations=["人工复核商标证据后再决定是否上架。"],
        missing_evidence=[],
        limitations=["本报告仅用于知识产权风险初筛和证据发现，不构成法律意见。"],
    )
    assert report.to_dict()["overall_verdict"] == "caution"
    assert report.to_dict()["evidence_items"][0]["evidence_id"] == "E1"


def test_trace_event_has_json_serializable_payload():
    event = TraceEvent(
        trace_id="trace-1",
        step="planner",
        event_type="llm_plan",
        payload={"tool_count": 2},
        timestamp="2026-08-03T00:00:00Z",
    )
    assert event.to_dict()["payload"] == {"tool_count": 2}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_v1_schema_contracts.py`

Expected exit code: non-zero.

Expected output must contain:

```text
ImportError
cannot import name 'EvidenceHit'
```

- [ ] **Step 3: Write minimal implementation**

Add `ImageAsset`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ImageAsset:
    image_id: str
    source_doc_id: str
    storage_path: str = ""
    caption: str = ""
    ocr_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.image_id.strip():
            raise ValueError("image_id must be non-empty")
        if not self.source_doc_id.strip():
            raise ValueError("source_doc_id must be non-empty")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dict")

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "source_doc_id": self.source_doc_id,
            "storage_path": self.storage_path,
            "caption": self.caption,
            "ocr_text": self.ocr_text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageAsset":
        return cls(**data)
```

Add `EvidenceHit`, `RiskVerdict`, `RiskScreeningReport`, `TraceEvent`, and `EvaluationRun` with `to_dict()` and `from_dict()` methods. Use only JSON-serializable primitives in `to_dict()`.

- [ ] **Step 4: Update existing schemas to carry images**

Modify `NormalizedDocument` and `EvidenceChunk` to accept `images: list[ImageAsset] = field(default_factory=list)`. In `from_dict()`, convert image dictionaries with `ImageAsset.from_dict()`.

- [ ] **Step 5: Export schemas**

Modify `src/crossborder_agentic_rag/schemas/__init__.py`:

```python
from crossborder_agentic_rag.schemas.documents import NormalizedDocument
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk, EvidenceHit
from crossborder_agentic_rag.schemas.evaluation import EvaluationRun
from crossborder_agentic_rag.schemas.images import ImageAsset
from crossborder_agentic_rag.schemas.reports import RiskScreeningReport, RiskVerdict
from crossborder_agentic_rag.schemas.traces import TraceEvent

__all__ = [
    "EvaluationRun",
    "EvidenceChunk",
    "EvidenceHit",
    "ImageAsset",
    "NormalizedDocument",
    "RiskScreeningReport",
    "RiskVerdict",
    "TraceEvent",
]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest -q tests/test_v1_schema_contracts.py`

Expected exit code: 0.

Expected output must contain:

```text
passed
```

- [ ] **Step 7: Commit**

```bash
git add src/crossborder_agentic_rag/schemas tests/test_v1_schema_contracts.py
git commit -m "feat: add v1 risk report schemas"
```

Expected output must contain:

```text
feat: add v1 risk report schemas
```

**Independent Acceptance:** The schema tests pass and existing parser tests still import `NormalizedDocument` and `EvidenceChunk`.

---

### Task 2: Config Loader and Plugin Registry

**Files:**
- Create: `src/crossborder_agentic_rag/config/registry.py`
- Modify: `src/crossborder_agentic_rag/config/settings.py`
- Create: `configs/app.yaml`
- Create: `configs/plugins.yaml`
- Test: `tests/test_v1_config_registry.py`

**Interfaces:**
- Consumes: no task-specific interfaces.
- Produces: `AppConfig`, `PluginConfig`, `load_app_config(path: str | Path) -> AppConfig`, `PluginRegistry`.
- Consumed by later tasks: LLM client, agentic runtime, MCP server, dashboard, evaluation runner.

**TDD Rhythm:** Do Step 1 first and change only `tests/test_v1_config_registry.py`. Do not add `registry.py`, `AppConfig`, or YAML examples until Step 2 fails for the expected missing symbol. Step 6 must rerun the same task test file and turn it green before the commit.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from crossborder_agentic_rag.config.registry import PluginRegistry
from crossborder_agentic_rag.config.settings import load_app_config


def test_load_app_config_reads_plugin_choices(tmp_path: Path):
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        "llm:\n"
        "  provider: template\n"
        "  model: template\n"
        "  disable_thinking: true\n"
        "retrieval:\n"
        "  default_mode: hybrid_rerank\n"
        "observability:\n"
        "  provider: local_jsonl\n",
        encoding="utf-8",
    )
    cfg = load_app_config(config_path)
    assert cfg.llm["disable_thinking"] is True
    assert cfg.retrieval["default_mode"] == "hybrid_rerank"


def test_plugin_registry_registers_and_builds_provider():
    registry = PluginRegistry()
    registry.register("llm", "template", lambda cfg: {"provider": cfg["provider"]})
    built = registry.build("llm", {"provider": "template"})
    assert built == {"provider": "template"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_v1_config_registry.py`

Expected exit code: non-zero.

Expected output must contain:

```text
ModuleNotFoundError
crossborder_agentic_rag.config.registry
```

- [ ] **Step 3: Write minimal implementation**

Add to `settings.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AppConfig:
    llm: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    observability: dict[str, Any] = field(default_factory=dict)
    mcp: dict[str, Any] = field(default_factory=dict)
    dashboard: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)


def load_app_config(path: str | Path) -> AppConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise TypeError("app config must be a mapping")
    return AppConfig(
        llm=dict(data.get("llm") or {}),
        retrieval=dict(data.get("retrieval") or {}),
        observability=dict(data.get("observability") or {}),
        mcp=dict(data.get("mcp") or {}),
        dashboard=dict(data.get("dashboard") or {}),
        evaluation=dict(data.get("evaluation") or {}),
    )
```

- [ ] **Step 4: Implement registry**

```python
from __future__ import annotations

from collections.abc import Callable
from typing import Any


class PluginRegistry:
    def __init__(self) -> None:
        self._builders: dict[tuple[str, str], Callable[[dict[str, Any]], Any]] = {}

    def register(self, category: str, name: str, builder: Callable[[dict[str, Any]], Any]) -> None:
        key = (category.strip(), name.strip())
        if not key[0] or not key[1]:
            raise ValueError("category and name must be non-empty")
        self._builders[key] = builder

    def build(self, category: str, config: dict[str, Any]) -> Any:
        name = str(config.get("provider") or config.get("name") or "").strip()
        key = (category.strip(), name)
        if key not in self._builders:
            raise KeyError(f"No plugin registered for {category}:{name}")
        return self._builders[key](config)
```

- [ ] **Step 5: Add config examples**

Create `configs/app.yaml`:

```yaml
llm:
  provider: template
  model: template
  disable_thinking: true
retrieval:
  default_mode: hybrid_rerank
  top_k: 8
  candidate_k: 50
observability:
  provider: local_jsonl
mcp:
  enabled: true
dashboard:
  enabled: true
evaluation:
  golden_set: eval/queries_small.jsonl
```

Create `configs/plugins.yaml`:

```yaml
llm:
  providers:
    - template
    - openai_compatible
embedding:
  providers:
    - fake
    - local
    - openai_compatible
reranker:
  providers:
    - noop
    - lexical
    - local
trace:
  providers:
    - local_jsonl
    - langfuse
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest -q tests/test_v1_config_registry.py`

Expected exit code: 0.

Expected output must contain:

```text
passed
```

- [ ] **Step 7: Commit**

```bash
git add src/crossborder_agentic_rag/config configs/app.yaml configs/plugins.yaml tests/test_v1_config_registry.py
git commit -m "feat: add v1 config registry"
```

Expected output must contain:

```text
feat: add v1 config registry
```

**Independent Acceptance:** Config files load without environment variables and the registry can build a registered provider by category and provider name.

---

### Task 3: Image Contract Compatibility in Ingestion

**Files:**
- Modify: `src/crossborder_agentic_rag/ingestion/io_utils.py`
- Modify: `src/crossborder_agentic_rag/ingestion/trademark_parser.py`
- Modify: `src/crossborder_agentic_rag/ingestion/patent_parser.py`
- Modify: `src/crossborder_agentic_rag/ingestion/litigation_parser.py`
- Modify: `src/crossborder_agentic_rag/ingestion/chunkers.py`
- Test: `tests/test_v1_schema_contracts.py`
- Test: `tests/test_stage2_parsers.py`
- Test: `tests/test_stage3_chunkers.py`

**Interfaces:**
- Consumes: `ImageAsset`, `NormalizedDocument.images`, `EvidenceChunk.images`.
- Produces: parser and chunker compatibility where all text-source documents and chunks emit `images=[]`.
- Consumed by later tasks: report builder, dashboard evidence browser, MCP evidence resources.

**TDD Rhythm:** Do Step 1 first and append only the specified tests. Do not update JSONL conversion or chunk propagation until Step 2 fails for the expected missing `images` contract. Step 5 must rerun the focused parser/chunker command and turn it green before the commit.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_v1_schema_contracts.py`:

```python
from pathlib import Path

from crossborder_agentic_rag.ingestion.io_utils import read_documents_jsonl, read_chunks_jsonl
from crossborder_agentic_rag.schemas import NormalizedDocument


def test_document_from_legacy_json_without_images():
    doc = NormalizedDocument.from_dict(
        {
            "doc_id": "patent:1",
            "source_type": "patent",
            "title": "Patent 1",
            "content": "Claim evidence",
            "metadata": {},
        }
    )
    assert doc.images == []


def test_read_fixture_chunks_have_images_field():
    chunks = read_chunks_jsonl(Path("tests/fixtures/agent/sample_chunks.jsonl"))
    assert chunks
    assert all(c.images == [] for c in chunks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_v1_schema_contracts.py::test_read_fixture_chunks_have_images_field`

Expected exit code: non-zero.

Expected output must contain:

```text
AttributeError
object has no attribute 'images'
```

- [ ] **Step 3: Write minimal implementation**

In `io_utils.py`, ensure `from_dict()` handles missing `images` through schema defaults. Do not manually insert image dictionaries in parser code.

- [ ] **Step 4: Update chunker propagation**

When building chunks from `NormalizedDocument`, pass `images=list(doc.images)` into `EvidenceChunk`. For current text sources this value is an empty list.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest -q tests/test_v1_schema_contracts.py tests/test_stage2_parsers.py tests/test_stage3_chunkers.py`

Expected exit code: 0.

Expected output must contain:

```text
passed
```

- [ ] **Step 6: Commit**

```bash
git add src/crossborder_agentic_rag/ingestion src/crossborder_agentic_rag/schemas tests/test_v1_schema_contracts.py
git commit -m "feat: preserve image contract through ingestion"
```

Expected output must contain:

```text
feat: preserve image contract through ingestion
```

**Independent Acceptance:** Existing XML/TSV/CSV fixtures still parse, and all normalized docs/chunks expose `images=[]`.

---

### Task 4: LLM Client Structured Output and Thinking-Disabled Options

**Files:**
- Modify: `src/crossborder_agentic_rag/llm/chat_client.py`
- Test: `tests/test_v1_llm_client.py`

**Interfaces:**
- Consumes: `AppConfig.llm` from Task 2.
- Produces: `BaseChatClient.complete_structured(messages, schema_name) -> dict[str, Any]`.
- Produces: `OpenAICompatibleChatClient` request metadata with `extra_body={"enable_thinking": False}` when `disable_thinking=True`.
- Consumed by later tasks: query rewriter, planner, report synthesis, RAGAS-compatible runner.

**TDD Rhythm:** Do Step 1 first and change only `tests/test_v1_llm_client.py`. Do not edit `chat_client.py` until Step 2 fails on the missing structured-output or thinking-disabled contract. Step 6 must rerun the same task test file and turn it green before the commit.

- [ ] **Step 1: Write the failing test**

```python
from crossborder_agentic_rag.llm.chat_client import (
    ChatResult,
    TemplateChatClient,
    build_chat_client,
)


def test_template_complete_structured_returns_dict():
    client = TemplateChatClient()
    result = client.complete_structured(
        [{"role": "user", "content": "plan trademark query"}],
        schema_name="planner",
    )
    assert isinstance(result, dict)
    assert result["schema_name"] == "planner"


def test_openai_compatible_client_records_disable_thinking():
    client = build_chat_client(
        provider="openai_compatible",
        api_key="EMPTY",
        base_url="http://example.invalid/v1",
        model="qwen-compatible",
        disable_thinking=True,
    )
    assert getattr(client, "disable_thinking") is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_v1_llm_client.py`

Expected exit code: non-zero.

Expected output must contain:

```text
AttributeError
complete_structured
```

- [ ] **Step 3: Write minimal implementation**

Use `Protocol` for `BaseChatClient` so concrete clients carry the runtime behavior. Add the exact structured-output method signature to the protocol:

```python
class BaseChatClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        pass

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema_name: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        pass
```

Implement in `TemplateChatClient`:

```python
def complete_structured(self, messages, schema_name, temperature=None, max_tokens=None):
    return {
        "schema_name": schema_name,
        "provider": self.provider,
        "model": self.model,
        "content": self.complete(messages, temperature=temperature, max_tokens=max_tokens).content,
    }
```

- [ ] **Step 4: Add disable_thinking to OpenAI-compatible client**

Extend `OpenAICompatibleChatClient.__init__()` with `disable_thinking: bool = True`. In `complete()`, when `disable_thinking` is true, pass `extra_body={"enable_thinking": False}` to `client.chat.completions.create()`. If the SDK raises `TypeError`, retry with `extra_body` omitted and return `ChatResult.raw` containing `{"disable_thinking_requested": True, "disable_thinking_applied": False}`.

- [ ] **Step 5: Implement JSON structured parsing**

In `OpenAICompatibleChatClient.complete_structured()`, call `complete()`, strip fenced JSON blocks, parse with `json.loads()`, and return a dict. If parsing fails, return:

```python
{
    "schema_name": schema_name,
    "error": "structured_output_parse_failed",
    "raw_content": result.content,
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest -q tests/test_v1_llm_client.py`

Expected exit code: 0.

Expected output must contain:

```text
passed
```

- [ ] **Step 7: Commit**

```bash
git add src/crossborder_agentic_rag/llm/chat_client.py tests/test_v1_llm_client.py
git commit -m "feat: add structured llm client support"
```

Expected output must contain:

```text
feat: add structured llm client support
```

**Independent Acceptance:** Template clients can produce structured dicts without network access, and OpenAI-compatible clients carry a thinking-disabled configuration.

---

### Task 5: Report Builder and Citation Audit

**Files:**
- Create: `src/crossborder_agentic_rag/reports/__init__.py`
- Create: `src/crossborder_agentic_rag/reports/builder.py`
- Create: `src/crossborder_agentic_rag/reports/citation_audit.py`
- Test: `tests/test_v1_report_builder.py`

**Interfaces:**
- Consumes: `EvidenceHit`, `RiskScreeningReport`, `RiskVerdict`.
- Produces: `build_risk_screening_report(query, target_markets, scope, evidence_hits, missing_evidence, trace_id) -> RiskScreeningReport`.
- Produces: `audit_report_citations(report: RiskScreeningReport) -> dict[str, Any]`.
- Consumed by later tasks: agentic runtime, MCP `query_ip_risk`, dashboard report page, evaluation runner.

**TDD Rhythm:** Do Step 1 first and change only `tests/test_v1_report_builder.py`. Do not create the `reports` package until Step 2 fails for the expected missing module. Step 5 must rerun the same task test file and turn it green before the commit.

- [ ] **Step 1: Write the failing test**

```python
from crossborder_agentic_rag.reports.builder import build_risk_screening_report
from crossborder_agentic_rag.reports.citation_audit import audit_report_citations
from crossborder_agentic_rag.schemas import EvidenceHit, RiskVerdict


def make_hit(source_type="trademark"):
    return EvidenceHit(
        evidence_id="E1",
        chunk_id="trademark:1:chunk:0",
        source_type=source_type,
        title="Trademark evidence",
        content="Registered mark evidence",
        citation="[trademark:1:chunk:0] Trademark evidence",
        rank=1,
        score=1.0,
        retrieval_mode="hybrid_rerank",
        tool_name="trademark_search_tool",
    )


def test_report_with_evidence_returns_caution():
    report = build_risk_screening_report(
        query="Can I sell this product in the US?",
        target_markets=["US"],
        scope=["trademark"],
        evidence_hits=[make_hit()],
        missing_evidence=[],
        trace_id="trace-1",
    )
    assert report.overall_verdict == RiskVerdict.CAUTION
    assert report.risk_cards["caution"] == 1
    assert report.evidence_items[0].evidence_id == "E1"


def test_report_without_evidence_is_insufficient():
    report = build_risk_screening_report(
        query="Can I sell this product in the US?",
        target_markets=["US"],
        scope=["trademark"],
        evidence_hits=[],
        missing_evidence=["trademark"],
        trace_id="trace-1",
    )
    assert report.overall_verdict == RiskVerdict.INSUFFICIENT_EVIDENCE


def test_citation_audit_rejects_missing_evidence_reference():
    report = build_risk_screening_report(
        query="q",
        target_markets=["US"],
        scope=["trademark"],
        evidence_hits=[make_hit()],
        missing_evidence=[],
        trace_id="trace-1",
    )
    result = audit_report_citations(report)
    assert result["valid_citation_rate"] == 1.0
    assert result["unsupported_claim_count"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_v1_report_builder.py`

Expected exit code: non-zero.

Expected output must contain:

```text
ModuleNotFoundError
crossborder_agentic_rag.reports
```

- [ ] **Step 3: Write minimal implementation**

Implement deterministic report logic:

```python
from datetime import datetime, timezone
from uuid import uuid4

from crossborder_agentic_rag.schemas import EvidenceHit, RiskScreeningReport, RiskVerdict


def build_risk_screening_report(query, target_markets, scope, evidence_hits, missing_evidence, trace_id):
    verdict = RiskVerdict.INSUFFICIENT_EVIDENCE if missing_evidence and not evidence_hits else RiskVerdict.CAUTION
    if evidence_hits and any(hit.source_type == "litigation" for hit in evidence_hits):
        verdict = RiskVerdict.NOT_RECOMMENDED
    if not evidence_hits and not missing_evidence:
        verdict = RiskVerdict.NO_RISK_FOUND
    risk_cards = {
        "no_risk_found": 1 if verdict == RiskVerdict.NO_RISK_FOUND else 0,
        "caution": 1 if verdict == RiskVerdict.CAUTION else 0,
        "not_recommended": 1 if verdict == RiskVerdict.NOT_RECOMMENDED else 0,
        "insufficient_evidence": 1 if verdict == RiskVerdict.INSUFFICIENT_EVIDENCE else 0,
    }
    summary_text = "命中知识产权风险信号，建议人工复核后再决定是否上架。" if evidence_hits else "证据不足，需补充索引或数据后再判断。"
    return RiskScreeningReport(
        report_id=f"report-{uuid4().hex}",
        trace_id=trace_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        product_profile={"query": query},
        target_markets=list(target_markets),
        screening_scope=list(scope),
        overall_verdict=verdict,
        country_summaries=[{"country": m, "verdict": verdict.value, "summary": summary_text} for m in target_markets],
        risk_cards=risk_cards,
        module_results=[{"module": s, "verdict": verdict.value, "evidence_ids": [h.evidence_id for h in evidence_hits if h.source_type == s]} for s in scope],
        evidence_items=list(evidence_hits),
        action_recommendations=["建议人工复核命中的商标、专利或诉讼证据后再决定是否上架。"],
        missing_evidence=list(missing_evidence),
        limitations=["本报告仅用于知识产权风险初筛和证据发现，不构成法律意见。"],
    )
```

- [ ] **Step 4: Implement citation audit**

Return `valid_citation_rate`, `citation_coverage`, and `unsupported_claim_count`. Count unsupported claims as zero only when every module result with a risk verdict has at least one matching evidence id.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest -q tests/test_v1_report_builder.py`

Expected exit code: 0.

Expected output must contain:

```text
passed
```

- [ ] **Step 6: Commit**

```bash
git add src/crossborder_agentic_rag/reports tests/test_v1_report_builder.py
git commit -m "feat: add structured risk report builder"
```

Expected output must contain:

```text
feat: add structured risk report builder
```

**Independent Acceptance:** A report can be built from evidence hits without invoking LLMs or retrievers, and citation audit returns deterministic metrics.

---

### Task 6: Agentic Query Rewrite and Planning Interfaces

**Files:**
- Create: `src/crossborder_agentic_rag/agentic/__init__.py`
- Create: `src/crossborder_agentic_rag/agentic/normalizer.py`
- Create: `src/crossborder_agentic_rag/agentic/rewriter.py`
- Create: `src/crossborder_agentic_rag/agentic/planner.py`
- Test: `tests/test_v1_agentic_runtime.py`

**Interfaces:**
- Consumes: `BaseChatClient.complete_structured`.
- Produces: `normalize_user_query(query: str, target_markets: list[str] | None) -> dict[str, Any]`.
- Produces: `rewrite_for_scenario(query: str, scope: list[str]) -> dict[str, str]`.
- Produces: `plan_tools(query: str, scope: list[str], llm: BaseChatClient | None) -> list[dict[str, Any]]`.
- Consumed by later tasks: dispatcher and runtime.

**TDD Rhythm:** Do Step 1 first and change only `tests/test_v1_agentic_runtime.py`. Do not create the `agentic` package until Step 2 fails for the expected missing package. Step 6 must rerun the listed focused tests and turn them green before the commit.

- [ ] **Step 1: Write the failing test**

```python
from crossborder_agentic_rag.agentic.normalizer import normalize_user_query
from crossborder_agentic_rag.agentic.planner import plan_tools
from crossborder_agentic_rag.agentic.rewriter import rewrite_for_scenario


def test_normalize_user_query_defaults_market_to_us():
    normalized = normalize_user_query("Can I sell a smart phone case?", None)
    assert normalized["query"] == "Can I sell a smart phone case?"
    assert normalized["target_markets"] == ["US"]


def test_rewrite_for_scenario_builds_domain_queries():
    rewritten = rewrite_for_scenario("smart phone case", ["trademark", "patent", "litigation"])
    assert "brand logo goods services smart phone case" in rewritten["trademark"]
    assert "technical features patent claims smart phone case" in rewritten["patent"]
    assert "litigation case asserted patent smart phone case" in rewritten["litigation"]


def test_plan_tools_uses_scope_without_llm():
    plan = plan_tools("smart phone case", ["trademark", "patent"], llm=None)
    assert [step["tool"] for step in plan] == ["trademark_search_tool", "patent_search_tool"]
    assert plan[0]["retrieval_mode"] == "hybrid_rerank"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_v1_agentic_runtime.py::test_normalize_user_query_defaults_market_to_us`

Expected exit code: non-zero.

Expected output must contain:

```text
ModuleNotFoundError
crossborder_agentic_rag.agentic
```

- [ ] **Step 3: Write minimal implementation**

```python
def normalize_user_query(query: str, target_markets: list[str] | None) -> dict[str, object]:
    cleaned = " ".join((query or "").split())
    if not cleaned:
        raise ValueError("query must be non-empty")
    markets = target_markets or ["US"]
    return {"query": cleaned, "target_markets": markets}
```

- [ ] **Step 4: Implement scenario rewriter**

```python
def rewrite_for_scenario(query: str, scope: list[str]) -> dict[str, str]:
    q = " ".join(query.split())
    out: dict[str, str] = {}
    if "trademark" in scope:
        out["trademark"] = f"brand logo goods services {q}"
    if "patent" in scope:
        out["patent"] = f"technical features patent claims {q}"
    if "litigation" in scope:
        out["litigation"] = f"litigation case asserted patent {q}"
    return out
```

- [ ] **Step 5: Implement planner**

Map scope values to tools:

```python
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
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest -q tests/test_v1_agentic_runtime.py::test_normalize_user_query_defaults_market_to_us tests/test_v1_agentic_runtime.py::test_rewrite_for_scenario_builds_domain_queries tests/test_v1_agentic_runtime.py::test_plan_tools_uses_scope_without_llm`

Expected exit code: 0.

Expected output must contain:

```text
passed
```

- [ ] **Step 7: Commit**

```bash
git add src/crossborder_agentic_rag/agentic tests/test_v1_agentic_runtime.py
git commit -m "feat: add agentic planning interfaces"
```

Expected output must contain:

```text
feat: add agentic planning interfaces
```

**Independent Acceptance:** Query normalization, scenario rewrite, and deterministic tool planning work without a model key.

---

### Task 7: Tool Dispatcher and Risk Screening Runtime

**Files:**
- Create: `src/crossborder_agentic_rag/agentic/dispatcher.py`
- Create: `src/crossborder_agentic_rag/agentic/evidence_gap.py`
- Create: `src/crossborder_agentic_rag/agentic/runtime.py`
- Modify: `tests/test_v1_agentic_runtime.py`

**Interfaces:**
- Consumes: `plan_tools`, `EvidenceHit`, `build_risk_screening_report`.
- Produces: `ToolDispatcher.run(action: dict[str, Any]) -> list[EvidenceHit]`.
- Produces: `RiskScreeningRuntime.run(query: str, target_markets: list[str] | None, scope: list[str]) -> RiskScreeningReport`.
- Consumed by later tasks: MCP `query_ip_risk`, CLI `query.py`, Streamlit services, evaluation runner.

**TDD Rhythm:** Do Step 1 first and append only the runtime test to `tests/test_v1_agentic_runtime.py`. Do not create `dispatcher.py`, `evidence_gap.py`, or `runtime.py` until Step 2 fails for the expected missing runtime module. Step 6 must rerun the runtime test file and turn it green before the commit.

- [ ] **Step 1: Write the failing test**

```python
from crossborder_agentic_rag.agentic.runtime import RiskScreeningRuntime
from crossborder_agentic_rag.schemas import EvidenceHit, RiskVerdict


class FakeDispatcher:
    def run(self, action):
        return [
            EvidenceHit(
                evidence_id="E1",
                chunk_id="trademark:1:chunk:0",
                source_type="trademark",
                title="Trademark evidence",
                content="Registered mark evidence",
                citation="[trademark:1:chunk:0] Trademark evidence",
                rank=1,
                score=1.0,
                retrieval_mode=action["retrieval_mode"],
                tool_name=action["tool"],
            )
        ]


def test_runtime_returns_structured_report():
    runtime = RiskScreeningRuntime(dispatcher=FakeDispatcher(), llm=None)
    report = runtime.run(
        query="Can I sell a smart phone case?",
        target_markets=["US"],
        scope=["trademark"],
    )
    assert report.overall_verdict == RiskVerdict.CAUTION
    assert report.evidence_items[0].tool_name == "trademark_search_tool"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_v1_agentic_runtime.py::test_runtime_returns_structured_report`

Expected exit code: non-zero.

Expected output must contain:

```text
ModuleNotFoundError
crossborder_agentic_rag.agentic.runtime
```

- [ ] **Step 3: Write minimal implementation**

```python
def find_missing_evidence(scope: list[str], hits) -> list[str]:
    present = {hit.source_type for hit in hits}
    return [item for item in scope if item not in present]
```

- [ ] **Step 4: Implement dispatcher adapter**

`ToolDispatcher` should accept optional `retriever`, `duckdb_store`, and `graph_retriever`. For v1 fixture tests, allow `None` backends and return an empty list. When `retriever` exists, convert returned chunks into `EvidenceHit`.

- [ ] **Step 5: Implement runtime**

```python
from crossborder_agentic_rag.agentic.evidence_gap import find_missing_evidence
from crossborder_agentic_rag.agentic.normalizer import normalize_user_query
from crossborder_agentic_rag.agentic.planner import plan_tools
from crossborder_agentic_rag.reports.builder import build_risk_screening_report


class RiskScreeningRuntime:
    def __init__(self, dispatcher, llm=None) -> None:
        self.dispatcher = dispatcher
        self.llm = llm

    def run(self, query: str, target_markets: list[str] | None = None, scope: list[str] | None = None):
        selected_scope = scope or ["trademark", "patent", "litigation"]
        normalized = normalize_user_query(query, target_markets)
        actions = plan_tools(str(normalized["query"]), selected_scope, llm=self.llm)
        hits = []
        for action in actions:
            hits.extend(self.dispatcher.run(action))
        missing = find_missing_evidence(selected_scope, hits)
        return build_risk_screening_report(
            query=str(normalized["query"]),
            target_markets=list(normalized["target_markets"]),
            scope=selected_scope,
            evidence_hits=hits,
            missing_evidence=missing,
            trace_id="trace-local",
        )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest -q tests/test_v1_agentic_runtime.py`

Expected exit code: 0.

Expected output must contain:

```text
passed
```

- [ ] **Step 7: Commit**

```bash
git add src/crossborder_agentic_rag/agentic tests/test_v1_agentic_runtime.py
git commit -m "feat: add risk screening runtime"
```

Expected output must contain:

```text
feat: add risk screening runtime
```

**Independent Acceptance:** The runtime can produce a report using a fake dispatcher and no external service.

---

### Task 8: Local JSONL and Langfuse Trace Adapters

**Files:**
- Create: `src/crossborder_agentic_rag/observability/__init__.py`
- Create: `src/crossborder_agentic_rag/observability/trace.py`
- Create: `src/crossborder_agentic_rag/observability/jsonl_trace.py`
- Create: `src/crossborder_agentic_rag/observability/langfuse_trace.py`
- Modify: `src/crossborder_agentic_rag/agentic/runtime.py`
- Test: `tests/test_v1_trace_adapters.py`

**Interfaces:**
- Consumes: `TraceEvent`.
- Produces: `TraceSink.record(event: TraceEvent) -> None`.
- Produces: `LocalJsonlTraceSink(path: str | Path)`.
- Produces: `LangfuseTraceSink(enabled: bool, fallback: TraceSink | None)`.
- Consumed by later tasks: MCP `get_trace`, dashboard trace page, evaluation artifacts.

**TDD Rhythm:** Do Step 1 first and change only `tests/test_v1_trace_adapters.py`. Do not create the `observability` package or modify runtime tracing until Step 2 fails for the expected missing module. Step 6 must rerun the trace and runtime test files and turn them green before the commit.

- [ ] **Step 1: Write the failing test**

```python
import json

from crossborder_agentic_rag.observability.jsonl_trace import LocalJsonlTraceSink
from crossborder_agentic_rag.schemas import TraceEvent


def test_local_jsonl_trace_sink_writes_event(tmp_path):
    path = tmp_path / "trace.jsonl"
    sink = LocalJsonlTraceSink(path)
    sink.record(
        TraceEvent(
            trace_id="trace-1",
            step="planner",
            event_type="tool_plan",
            payload={"tool_count": 1},
            timestamp="2026-08-03T00:00:00Z",
        )
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["event_type"] == "tool_plan"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_v1_trace_adapters.py`

Expected exit code: non-zero.

Expected output must contain:

```text
ModuleNotFoundError
crossborder_agentic_rag.observability
```

- [ ] **Step 3: Write minimal implementation**

```python
from pathlib import Path


class LocalJsonlTraceSink:
    def __init__(self, path):
        self.path = Path(path)

    def record(self, event):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Implement Langfuse fallback sink**

`LangfuseTraceSink.record()` should call fallback when disabled or when Langfuse import fails. Store failures in fallback payload with `event_type="trace_backend_error"`.

- [ ] **Step 5: Wire runtime trace events**

Add optional `trace_sink` to `RiskScreeningRuntime`. Record events for `normalize_query`, `plan_tools`, `tool_call`, `evidence_gap`, and `report`.

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest -q tests/test_v1_trace_adapters.py tests/test_v1_agentic_runtime.py`

Expected exit code: 0.

Expected output must contain:

```text
passed
```

- [ ] **Step 7: Commit**

```bash
git add src/crossborder_agentic_rag/observability src/crossborder_agentic_rag/agentic/runtime.py tests/test_v1_trace_adapters.py tests/test_v1_agentic_runtime.py
git commit -m "feat: add trace adapters"
```

Expected output must contain:

```text
feat: add trace adapters
```

**Independent Acceptance:** Local JSONL trace writes events without Langfuse installed, and the runtime can run with a trace sink.

---

### Task 9: MCP Tool and Resource Layer

**Files:**
- Create: `src/crossborder_agentic_rag/mcp_server/__init__.py`
- Create: `src/crossborder_agentic_rag/mcp_server/tools.py`
- Create: `src/crossborder_agentic_rag/mcp_server/resources.py`
- Create: `src/crossborder_agentic_rag/mcp_server/server.py`
- Create: `scripts/run_mcp_server.py`
- Modify: `pyproject.toml`
- Test: `tests/test_v1_mcp_tools.py`

**Interfaces:**
- Consumes: `RiskScreeningRuntime.run()`, `RiskScreeningReport.to_dict()`, trace artifacts.
- Produces: `query_ip_risk_tool(payload: dict[str, Any], runtime) -> dict[str, Any]`.
- Produces: `search_evidence_tool(payload: dict[str, Any], dispatcher) -> dict[str, Any]`.
- Produces: `get_trace_resource(trace_id: str, trace_dir: Path) -> dict[str, Any]`.
- Consumed by external MCP clients and dashboard smoke checks.

**TDD Rhythm:** Do Step 1 first and change only `tests/test_v1_mcp_tools.py`. Do not create the `mcp_server` package or scripts until Step 2 fails for the expected missing module. Step 6 must rerun the MCP tool test file and turn it green before the commit.

- [ ] **Step 1: Write the failing test**

```python
from crossborder_agentic_rag.mcp_server.tools import query_ip_risk_tool
from crossborder_agentic_rag.schemas import RiskScreeningReport, RiskVerdict


class FakeRuntime:
    def run(self, query, target_markets=None, scope=None):
        return RiskScreeningReport(
            report_id="report-1",
            trace_id="trace-1",
            created_at="2026-08-03T00:00:00Z",
            product_profile={"query": query},
            target_markets=target_markets or ["US"],
            screening_scope=scope or ["trademark"],
            overall_verdict=RiskVerdict.INSUFFICIENT_EVIDENCE,
            country_summaries=[],
            risk_cards={"no_risk_found": 0, "caution": 0, "not_recommended": 0, "insufficient_evidence": 1},
            module_results=[],
            evidence_items=[],
            action_recommendations=[],
            missing_evidence=["trademark"],
            limitations=["本报告仅用于知识产权风险初筛和证据发现，不构成法律意见。"],
        )


def test_query_ip_risk_tool_returns_structured_content():
    response = query_ip_risk_tool(
        {"query": "Can I sell this?", "target_markets": ["US"], "scope": ["trademark"]},
        runtime=FakeRuntime(),
    )
    assert response["structuredContent"]["overall_verdict"] == "insufficient_evidence"
    assert response["content"][0]["type"] == "text"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_v1_mcp_tools.py`

Expected exit code: non-zero.

Expected output must contain:

```text
ModuleNotFoundError
crossborder_agentic_rag.mcp_server
```

- [ ] **Step 3: Write minimal implementation**

`query_ip_risk_tool()` must validate `query`, call runtime, and return:

```python
{
    "structuredContent": report.to_dict(),
    "content": [
        {
            "type": "text",
            "text": f"{report.overall_verdict.value}: {len(report.evidence_items)} evidence items",
        }
    ],
}
```

- [ ] **Step 4: Implement resources**

`get_trace_resource(trace_id, trace_dir)` reads `reports/traces/{trace_id}.jsonl` and returns `{"trace_id": trace_id, "events": events}`. If missing, return `{"trace_id": trace_id, "error": "TRACE_NOT_FOUND"}`.

- [ ] **Step 5: Add server entry**

Create `scripts/run_mcp_server.py` that imports `crossborder_agentic_rag.mcp_server.server` and calls `main()`. The server module can expose a `main()` that prints a clear message when the optional MCP package is not installed.

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest -q tests/test_v1_mcp_tools.py`

Expected exit code: 0.

Expected output must contain:

```text
passed
```

- [ ] **Step 7: Commit**

```bash
git add src/crossborder_agentic_rag/mcp_server scripts/run_mcp_server.py pyproject.toml tests/test_v1_mcp_tools.py
git commit -m "feat: add mcp tool contracts"
```

Expected output must contain:

```text
feat: add mcp tool contracts
```

**Independent Acceptance:** MCP tool contract tests pass without launching a server or requiring network access.

---

### Task 10: Streamlit Dashboard Service Layer

**Files:**
- Create: `src/crossborder_agentic_rag/dashboard/__init__.py`
- Create: `src/crossborder_agentic_rag/dashboard/services.py`
- Create: `src/crossborder_agentic_rag/dashboard/app.py`
- Create: `scripts/run_dashboard.py`
- Modify: `pyproject.toml`
- Test: `tests/test_v1_dashboard_services.py`

**Interfaces:**
- Consumes: `RiskScreeningReport`, trace resources, evaluation artifacts.
- Produces: `summarize_report(report: RiskScreeningReport) -> dict[str, Any]`.
- Produces: `load_eval_summary(path: str | Path) -> dict[str, Any]`.
- Produces: Streamlit app import that does not run queries at import time.
- Consumed by manual dashboard execution and smoke tests.

**TDD Rhythm:** Do Step 1 first and change only `tests/test_v1_dashboard_services.py`. Do not create the `dashboard` package or Streamlit script until Step 2 fails for the expected missing service module. Step 6 must rerun the dashboard service test file and turn it green before the commit.

- [ ] **Step 1: Write the failing test**

```python
from crossborder_agentic_rag.dashboard.services import summarize_report
from crossborder_agentic_rag.schemas import RiskScreeningReport, RiskVerdict


def test_summarize_report_counts_cards():
    report = RiskScreeningReport(
        report_id="report-1",
        trace_id="trace-1",
        created_at="2026-08-03T00:00:00Z",
        product_profile={"query": "phone case"},
        target_markets=["US"],
        screening_scope=["trademark"],
        overall_verdict=RiskVerdict.CAUTION,
        country_summaries=[],
        risk_cards={"no_risk_found": 0, "caution": 1, "not_recommended": 0, "insufficient_evidence": 0},
        module_results=[],
        evidence_items=[],
        action_recommendations=[],
        missing_evidence=[],
        limitations=[],
    )
    summary = summarize_report(report)
    assert summary["report_id"] == "report-1"
    assert summary["cards"]["caution"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_v1_dashboard_services.py`

Expected exit code: non-zero.

Expected output must contain:

```text
ModuleNotFoundError
crossborder_agentic_rag.dashboard
```

- [ ] **Step 3: Write minimal implementation**

`summarize_report()` returns a JSON-friendly dict with `report_id`, `trace_id`, `overall_verdict`, `cards`, `target_markets`, and `evidence_count`.

- [ ] **Step 4: Implement app import guard**

In `dashboard/app.py`, define `main()` and import Streamlit inside `main()`:

```python
def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="IP Risk Agentic RAG", layout="wide")
    st.title("基于 Agentic RAG 的跨境电商知识产权风险初筛系统")
```

- [ ] **Step 5: Add script**

`scripts/run_dashboard.py` calls `crossborder_agentic_rag.dashboard.app.main()`.

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest -q tests/test_v1_dashboard_services.py`

Expected exit code: 0.

Expected output must contain:

```text
passed
```

- [ ] **Step 7: Commit**

```bash
git add src/crossborder_agentic_rag/dashboard scripts/run_dashboard.py pyproject.toml tests/test_v1_dashboard_services.py
git commit -m "feat: add dashboard service layer"
```

Expected output must contain:

```text
feat: add dashboard service layer
```

**Independent Acceptance:** Dashboard services can summarize a report without Streamlit installed, and app import does not trigger runtime work.

---

### Task 11: Evaluation Runner for Retrieval, Citation, Agent, and RAGAS Artifacts

**Files:**
- Create: `src/crossborder_agentic_rag/evaluation/citation_metrics.py`
- Create: `src/crossborder_agentic_rag/evaluation/agent_metrics.py`
- Create: `src/crossborder_agentic_rag/evaluation/eval_runner.py`
- Create: `scripts/evaluate.py`
- Test: `tests/test_v1_evaluation_runner.py`

**Interfaces:**
- Consumes: `RiskScreeningRuntime`, `RiskScreeningReport`, `audit_report_citations`.
- Produces: `run_fixture_evaluation(eval_rows: list[dict[str, Any]], runtime) -> EvaluationRun`.
- Produces: JSON files `summary.json`, `citation_audit.json`, `agent_metrics.json`.
- Consumed by dashboard eval panel and MCP `get_eval_report`.

**TDD Rhythm:** Do Step 1 first and change only `tests/test_v1_evaluation_runner.py`. Do not create `eval_runner.py`, new metrics modules, or `scripts/evaluate.py` until Step 2 fails for the expected missing eval runner. Step 7 must rerun the evaluation test file and turn it green before the commit.

- [ ] **Step 1: Write the failing test**

```python
from crossborder_agentic_rag.evaluation.eval_runner import run_fixture_evaluation
from crossborder_agentic_rag.schemas import RiskScreeningReport, RiskVerdict


class FakeRuntime:
    def run(self, query, target_markets=None, scope=None):
        return RiskScreeningReport(
            report_id="report-1",
            trace_id="trace-1",
            created_at="2026-08-03T00:00:00Z",
            product_profile={"query": query},
            target_markets=target_markets or ["US"],
            screening_scope=scope or ["trademark"],
            overall_verdict=RiskVerdict.INSUFFICIENT_EVIDENCE,
            country_summaries=[],
            risk_cards={"no_risk_found": 0, "caution": 0, "not_recommended": 0, "insufficient_evidence": 1},
            module_results=[],
            evidence_items=[],
            action_recommendations=[],
            missing_evidence=["trademark"],
            limitations=[],
        )


def test_run_fixture_evaluation_returns_summary():
    run = run_fixture_evaluation(
        [{"id": "Q1", "query": "Can I sell this?", "target_markets": ["US"], "scope": ["trademark"]}],
        runtime=FakeRuntime(),
    )
    assert run.run_id
    assert run.summary["n"] == 1
    assert run.summary["insufficient_evidence"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_v1_evaluation_runner.py`

Expected exit code: non-zero.

Expected output must contain:

```text
ModuleNotFoundError
crossborder_agentic_rag.evaluation.eval_runner
```

- [ ] **Step 3: Write minimal implementation**

Wrap `audit_report_citations()` so evaluation can aggregate `valid_citation_rate`, `citation_coverage`, and `unsupported_claim_count`.

- [ ] **Step 4: Implement agent metrics**

For each report, compute:

```python
{
    "tool_failure_rate": 0.0,
    "missing_evidence_count": len(report.missing_evidence),
    "evidence_count": len(report.evidence_items),
}
```

- [ ] **Step 5: Implement evaluation runner**

`run_fixture_evaluation()` loops rows, calls runtime, counts verdicts, aggregates citation and agent metrics, and returns `EvaluationRun`.

- [ ] **Step 6: Add CLI**

`scripts/evaluate.py` accepts `--eval-file` and `--output-dir`. It loads JSONL rows, runs fixture-safe runtime when no real backend is configured, and writes `summary.json`.

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest -q tests/test_v1_evaluation_runner.py`

Expected exit code: 0.

Expected output must contain:

```text
passed
```

- [ ] **Step 8: Commit**

```bash
git add src/crossborder_agentic_rag/evaluation scripts/evaluate.py tests/test_v1_evaluation_runner.py
git commit -m "feat: add v1 evaluation runner"
```

Expected output must contain:

```text
feat: add v1 evaluation runner
```

**Independent Acceptance:** Fixture evaluation returns an `EvaluationRun` and can be extended to real retrieval/RAGAS runs.

---

### Task 12: Stable Query CLI

**Files:**
- Create: `scripts/query.py`
- Modify: `README.md`
- Test: `tests/test_v1_cli_contracts.py`

**Interfaces:**
- Consumes: `RiskScreeningRuntime.run()`, `RiskScreeningReport.to_dict()`.
- Produces: CLI command `python scripts/query.py "query" --target-market US --scope trademark --output-json`.
- Consumed by users, MCP smoke scripts, and dashboard local demos.

**TDD Rhythm:** Do Step 1 first and change only `tests/test_v1_cli_contracts.py`. Do not create `scripts/query.py` or update README until Step 2 fails for the expected missing CLI script. Step 6 must rerun the CLI contract test and turn it green before the commit.

- [ ] **Step 1: Write the failing test**

```python
import json
import subprocess
import sys


def test_query_cli_outputs_report_json():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/query.py",
            "Can I sell this phone case?",
            "--target-market",
            "US",
            "--scope",
            "trademark",
            "--output-json",
        ],
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["target_markets"] == ["US"]
    assert data["overall_verdict"] in {"no_risk_found", "caution", "not_recommended", "insufficient_evidence"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_v1_cli_contracts.py`

Expected exit code: non-zero.

Expected output must contain:

```text
AssertionError
scripts/query.py
```

- [ ] **Step 3: Write minimal implementation**

Use `argparse` with:

```text
query positional
--target-market action append default ["US"]
--scope action append default ["trademark", "patent", "litigation"]
--output-json boolean
```

- [ ] **Step 4: Wire runtime**

Instantiate `RiskScreeningRuntime(dispatcher=ToolDispatcher(), llm=None)` for fixture-safe output when no backends are configured. Print `json.dumps(report.to_dict(), ensure_ascii=False)` when `--output-json` is set.

- [ ] **Step 5: Update README command block**

Add:

```bash
python scripts/query.py "Can I sell this phone case in the US?" --target-market US --scope trademark --output-json
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest -q tests/test_v1_cli_contracts.py`

Expected exit code: 0.

Expected output must contain:

```text
passed
```

- [ ] **Step 7: Commit**

```bash
git add scripts/query.py README.md tests/test_v1_cli_contracts.py
git commit -m "feat: add stable risk query cli"
```

Expected output must contain:

```text
feat: add stable risk query cli
```

**Independent Acceptance:** The stable query CLI returns a JSON `RiskScreeningReport` without external services.

---

### Task 13: Full Fixture Gate and Documentation Alignment

**Files:**
- Modify: `DEV_SPEC.md`
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`
- Test: existing tests plus new v1 tests

**Interfaces:**
- Consumes: all previous task interfaces.
- Produces: CI gate that includes v1 contract tests and existing fixture tests.
- Consumed by maintainers before publishing releases.

- [ ] **Step 1: Add CI test list**

In `.github/workflows/ci.yml`, keep existing compile and pytest commands. If optional extras are not installed, avoid integration tests requiring Milvus, Langfuse, Streamlit server launch, or MCP server launch.

- [ ] **Step 2: Add README project identity**

README top section should state:

```markdown
# 基于 Agentic RAG 的跨境电商知识产权风险初筛系统

This project builds a single-turn, tool-planning Agentic RAG system for preliminary cross-border e-commerce IP risk screening.
```

- [ ] **Step 3: Add README stable workflows**

README should show:

```bash
python scripts/query.py "Can I sell this phone case in the US?" --target-market US --output-json
python scripts/evaluate.py --eval-file eval/queries_small.jsonl --output-dir reports/eval/demo
python scripts/run_mcp_server.py --help
python scripts/run_dashboard.py --help
```

- [ ] **Step 4: Run focused v1 gate**

Run:

```bash
python -m compileall -q src scripts
pytest -q tests/test_v1_schema_contracts.py tests/test_v1_config_registry.py tests/test_v1_llm_client.py tests/test_v1_report_builder.py tests/test_v1_agentic_runtime.py tests/test_v1_trace_adapters.py tests/test_v1_mcp_tools.py tests/test_v1_dashboard_services.py tests/test_v1_evaluation_runner.py tests/test_v1_cli_contracts.py
```

Expected: PASS.

- [ ] **Step 5: Run existing fixture gate**

Run:

```bash
pytest -q tests/test_stage1_core_interfaces.py tests/test_stage2_parsers.py tests/test_stage3_chunkers.py tests/test_stage6_agent_workflow.py tests/test_stage8_end_to_end.py
```

Expected: PASS.

- [ ] **Step 6: Run full suite if the repository quality gate is already aligned**

Run: `pytest -q`

Expected: PASS after runtime artifact path contracts are aligned with the v1 project structure.

- [ ] **Step 7: Commit**

```bash
git add DEV_SPEC.md README.md .github/workflows/ci.yml tests
git commit -m "docs: align v1 implementation workflow"
```

**Independent Acceptance:** v1 contract tests pass, key fixture tests pass, and README documents the stable workflows defined by `DEV_SPEC.md`.

---

## Coverage Map

| DEV_SPEC Area | Implemented By |
|---|---|
| Stable domain objects | Task 1 |
| ImageAsset contract and `images=[]` compatibility | Task 1, Task 3 |
| Config-driven pluggability | Task 2 |
| Thinking-disabled LLM layer | Task 4 |
| Structured risk screening report | Task 5 |
| Query normalization and rewrite | Task 6 |
| Single-agent tool planning runtime | Task 6, Task 7 |
| Evidence gap checking | Task 7 |
| Traceability and Langfuse/local fallback | Task 8 |
| MCP tools and resources | Task 9 |
| Streamlit workbench service layer | Task 10 |
| Evaluation framework | Task 11 |
| Stable CLI workflows | Task 12 |
| Documentation and CI alignment | Task 13 |

## Execution Notes

- Start with Task 1 and move sequentially through Task 13. Later tasks rely on interfaces introduced earlier.
- Tasks 1-12 must follow the internal TDD rhythm exactly: write the specified failing test, run the exact failing command, implement the minimal code, rerun the exact green command, then commit. Task 13 is the final documentation and fixture gate rather than a feature TDD task.
- Each task has its own test command and commit step. Do not combine tasks unless the same reviewer will validate the combined diff.
- Use fake, template, or fixture backends in unit tests. Real Milvus, real LLM, real Langfuse, Streamlit browser launch, and RAGAS generation runs stay optional integration checks.
- When adding optional dependencies, keep imports inside functions that need them so default package import remains clean.
- Do not expose chain-of-thought in reports, MCP responses, Streamlit UI, local traces, or evaluation artifacts.
