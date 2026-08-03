# Developer Specification (DEV_SPEC): 基于 Agentic RAG 的跨境电商知识产权风险初筛系统

> 版本：0.1
> 日期：2026-08-03
> 状态：项目规格草案，等待用户审阅
> 适用仓库：`dyy21zyy/Agentic-RAG-CrossBorder-Marketplace`
> 系统目标：构建一个单轮、工具规划式、MCP 可调用、Streamlit 可视化、Langfuse 可观测、评测驱动的跨境电商知识产权风险初筛系统，核心证据域覆盖 trademark、patent、litigation，并为图片与文档多模态扩展预留统一契约。

## 目录

- 项目概述
- 核心特点
- 技术选型
- 目标输出：风险初筛报告
- 系统架构与模块设计
- 全链路可插拔架构
- MCP 生态集成
- Streamlit 可视化管理平台
- 测试与评估方案
- 项目目录结构
- 项目排期
- 可扩展性与未来展望
- 安全边界与非目标
- 验收标准
- 关键设计决策记录

---

## 1. 项目概述

本项目的正式定位是：

> **基于 Agentic RAG 的跨境电商知识产权风险初筛系统**
> An Agentic RAG-based preliminary IP risk screening system for cross-border e-commerce.

本项目面向跨境电商卖家、合规人员、产品选品人员和知识产权风险分析人员，目标是把商标、专利和诉讼证据组织成可检索、可引用、可评估的风险初筛报告。系统以 Agentic RAG 为核心方法，通过 LLM 工具规划、查询改写、混合检索、结构化查库、GraphRAG 关系扩展、证据缺口检查和报告生成，将原本分散在不同公开数据源中的知识产权信息转化为可复盘的初筛结论。

### 1.1 设计理念 (Design Philosophy)

本项目的设计理念是：

> **证据优先、工具规划、可观测、可扩展。**

跨境电商知识产权风险初筛不是普通闲聊问答。系统需要回答的不是“模型认为是否有风险”，而是“哪些商标、专利或诉讼证据支持某个风险信号”。因此，本项目采用证据驱动架构：LLM 负责理解问题、规划工具、改写查询和组织报告，Python runtime 负责执行检索、约束边界、检查引用和记录 trace。

v1 的核心目标不是构建一个通用 RAG demo，而是形成一个可持续迭代的专业工程基座：

- 以 trademark / patent / litigation 为核心证据域，保持跨境电商 IP 风险初筛的业务深度。
- 以 MCP 作为外部 AI clients 的标准接入方式，让系统可以被 Claude Desktop、Cursor、Copilot/Codex 类工具调用。
- 以 Streamlit 提供本地风险初筛工作台，展示报告、证据、trace、评测和插件状态。
- 以 Langfuse + local JSONL trace 记录完整 agentic workflow，避免黑盒式回答。
- 以本地 golden set、retrieval metrics、citation audit 和 RAGAS 形成可复现评测闭环。
- 以可插拔架构支持未来模型、向量库、Loader、Reranker、Evaluator 和多模态能力替换。

### 1.2 核心证据范围

系统围绕商品上架前的知识产权风险初筛，检索并组织以下证据：

- 商标证据：USPTO trademark XML 中的 word mark、serial number、registration number、Nice class、goods/services、owner、status 等字段。
- 专利证据：PatentsView TSV 中的 patent metadata、claim-level text、claim number、claim dependency 等字段。
- 诉讼证据：patent litigation docket CSV 中的 cases、parties、documents、asserted patents、court、filing date 等字段。
- 未来扩展证据：平台政策、PDF/HTML/Markdown 文档、外观专利图片、版权相关素材、商品图或 logo 图。

### 1.3 使用场景

系统应帮助用户回答：

- 在目标市场中是否存在明显的商标、专利或诉讼风险信号？
- 哪些证据支持该风险信号？
- 哪些证据缺失或仍需要人工复核？
- 对未上架、已上架、无法更换设计等场景，应采取哪些保守行动建议？

### 1.4 稳定领域对象

为保证系统长期可演进，核心领域对象应保持稳定：

- `Document`
- `EvidenceChunk`
- `EvidenceHit`
- `ImageAsset`
- `RiskScreeningReport`
- `TraceEvent`
- `EvaluationRun`

技术组件可以替换，但领域语义不应漂移。

---

## 2. 核心特点

本项目的核心特点围绕“专业证据域 + Agentic RAG 工作流 + 可插拔工程化 + 可观测评测闭环”展开。与普通 RAG demo 相比，本系统强调证据可追溯、工具调用可解释、报告结构可复用、评测结果可复现。

### 2.1 Agentic RAG 工作流设计亮点

本项目使用 "Agentic RAG" 一词时，指的是单轮工具规划式 RAG workflow。v1 agentic capability 包含：

1. **Query Normalization**
   对用户输入进行基础清洗、语言规范化、目标市场识别和风险初筛场景识别。

2. **Scenario-aware Query Rewrite**
   将用户自然语言问题改写为适合知识产权风险初筛的检索表达，例如从“这个产品能不能在美国卖”改写为商标、专利、诉讼三个检索子问题。

3. **LLM Tool Planning**
   由 LLM planner 根据问题类型决定需要调用哪些工具，例如 `trademark_search_tool`、`patent_search_tool`、`litigation_search_tool`、`duckdb_lookup_tool`、`graph_rag_tool`。

4. **Tool-specific Query Rewrite**
   针对不同工具生成不同查询。例如同一个商品描述，对商标工具应强调 brand/logo/goods-services，对专利工具应强调 technical feature/claim language，对诉讼工具应强调 company/patent/case/docket。

5. **Retrieval Mode Selection**
   planner 可以为每个工具选择检索模式：`bm25_only`、`dense_only`、`hybrid_rrf`、`hybrid_rerank`、`duckdb_lookup`、`graph_rag`。

6. **Evidence Sufficiency Checking**
   检查当前证据是否覆盖报告所需的商标、专利、诉讼、结构化字段和引用。

7. **Bounded Follow-up Query Rewrite**
   当证据不足时，系统可以在最大迭代次数内生成 follow-up query。该过程必须有上限，避免无限检索。

8. **Structured Report Synthesis**
   最终生成结构化 `RiskScreeningReport`，而不是自由文本回答。

9. **Traceable Tool Execution**
   每个 planner 输出、query rewrite、tool call、retrieval result、rerank result、evidence gap、final report 都必须进入 trace。

### 2.2 多阶段证据检索策略

系统采用多阶段证据召回与精排机制，针对知识产权初筛中的不同问题类型选择不同检索路径：

- **结构化精确查找**：通过 DuckDB 查询 trademark registration、patent number、case number 等确定性字段。
- **稀疏关键词召回**：通过 BM25 捕捉品牌词、专利号、案号、商品类别等精确匹配信号。
- **稠密语义召回**：通过 embedding 检索产品功能描述、专利 claim 改写、诉讼摘要等语义相似内容。
- **RRF 融合排序**：将 sparse 和 dense 结果融合，降低单一路径漏召回风险。
- **Rerank 精排**：对候选证据进行二阶段重排，提高最终报告引用证据的相关性。
- **GraphRAG 关系扩展**：围绕 company、trademark、patent、case 等实体扩展多跳关系证据。

### 2.3 报告型输出

系统输出结构化 `RiskScreeningReport`，而不是自由文本 QA。报告包含检测范围、目标市场、风险统计卡片、国家级总结、行动建议、分模块详情、证据引用、缺失证据和 trace 信息，可直接支撑 Streamlit 页面和 MCP structuredContent。

### 2.4 全链路可观测与可评估

系统对 query normalization、scenario rewrite、LLM planner、tool-specific rewrite、tool call、retrieval、rerank、evidence gap、report synthesis 全链路记录 trace。Langfuse 用于交互式观察，本地 JSONL trace 用于离线和 CI 环境。评测体系覆盖 retrieval、citation、agent planning、report quality 和 RAGAS generation metrics。

### 2.5 多模态扩展预留

当前核心数据源是 XML/TSV/CSV，v1 不强行引入图片检索。但所有 Document/Chunk schema 都预留 `ImageAsset` 和 `images=[]`，未来 PDF/HTML/Markdown、外观专利图、版权素材或商品图可以通过 OCR/caption 进入统一文本检索链路。

### 2.6 Thinking 模式关闭原则

对于 Qwen、OpenAI-compatible 或其他可能启用 thinking/reasoning trace 的模型，系统应在 LLM provider 层统一注入关闭参数。例如：

```yaml
llm:
  provider: openai_compatible
  model: qwen-compatible-model
  disable_thinking: true
  temperature: 0
```

如果某个 provider 不支持 `enable_thinking=false`，适配器应记录该事实，并保证不将 chain-of-thought 输出进入最终报告、MCP response、Streamlit 页面或日志。

---

## 3. 技术选型

本项目技术选型遵循本地优先、可插拔、可观测和评测友好的原则。核心实现语言保持 Python，优先复用当前仓库已经具备测试基础的 DuckDB、BM25、Milvus、NetworkX GraphRAG、RRF、Reranker 和 RAGAS 评测能力。

### 3.1 数据摄取与规范化

当前核心数据源是 XML、TSV、CSV，因此 v1 不以 PDF loader 为主线，而是先把现有结构化/半结构化数据源规范化为稳定中间层：

- **Trademark Loader**：解析 USPTO trademark XML，输出 word mark、serial number、registration number、Nice class、goods/services、owner、status 等字段。
- **Patent Loader**：解析 PatentsView TSV，输出 patent metadata、claim number、claim text、claim dependency 等 claim-level 证据。
- **Litigation Loader**：解析 patent litigation docket CSV，输出 case、party、document、asserted patent、court、filing date 等诉讼证据。
- **Future Document Loader**：预留 PDF、HTML、Markdown、policy 文档和图片资产 loader。

所有 Loader 输出统一 `Document`，再经过 source-aware chunker 生成 `EvidenceChunk`。即使当前 XML/TSV/CSV 没有图片，schema 也应稳定包含 `images=[]`。

### 3.2 存储与索引

系统采用多存储协同，而不是单一向量库承载所有查询：

- **DuckDB**：用于精确结构化查找，如 trademark registration、patent number、case number。
- **BM25**：用于关键词、标识符、品牌词、案号等 lexical recall。
- **Milvus / Vector Store**：用于 dense semantic retrieval，支持产品功能描述和 claim paraphrase。
- **NetworkX GraphRAG**：用于 company、trademark、patent、case 之间的实体关系扩展。
- **Local JSONL artifacts**：用于 trace、evaluation outputs、offline report replay。

这种组合比单纯 vector search 更适合知识产权风险初筛，因为该场景同时需要精确编号、字段过滤、语义召回和关系扩展。

### 3.3 LLM、Embedding 与 Rerank

LLM、Embedding 和 Rerank 均通过统一接口接入：

- **LLM**：OpenAI-compatible API 为 v1 默认适配目标，同时预留 Azure OpenAI、Ollama、vLLM、LM Studio、DeepSeek、Anthropic 等后端。Qwen/OpenAI-compatible thinking 模式默认关闭。
- **Embedding**：支持 OpenAI/Azure embedding、本地 sentence-transformers、BGE/E5/GTE/Qwen embedding；Fake embedding 只用于测试和 smoke run。
- **Reranker**：支持 noop、lexical baseline、本地 BGE Cross-Encoder，未来预留 Cohere Rerank 和 LLM Rerank。

模型层必须记录 provider、model、维度、耗时、token usage、错误和 fallback，供 Langfuse、Streamlit 和 evaluation artifacts 使用。

### 3.4 Agentic Runtime

Agentic Runtime 采用单 Agent 工具规划型：

```text
User Query
-> Query Normalization
-> Scenario-aware Rewrite
-> LLM Planner
-> Tool-specific Query Rewrite
-> Tool Dispatch
-> Evidence Gap Check
-> Bounded Follow-up Rewrite
-> Final Rerank
-> RiskScreeningReport
```

该设计保持 agentic 能力，同时避免 v1 过早引入多 Agent 协作、长期 memory 或自治任务执行。

### 3.5 外部接口与可视化

- **MCP Server**：作为 AI clients 的主入口，暴露 `query_ip_risk`、`search_evidence`、`lookup_structured_record`、`list_sources`、`get_trace`、`get_eval_report`。
- **Streamlit Dashboard**：作为本地风险初筛工作台，展示报告、证据、trace、评测和插件状态。
- **CLI**：作为可复现批处理和评测入口，保留少量稳定命令。

### 3.6 可观测与评测

- **Langfuse**：记录 LLM planner、query rewrite、tool calls、retrieval、rerank 和 report synthesis。
- **Local JSONL Trace**：作为 Langfuse 不可用时的 fallback。
- **RAGAS**：用于 generation quality 评估，但不替代 retrieval/citation 指标。
- **本地评测指标**：HitRate@k、Recall@k、MRR、nDCG、ValidCitationRate、UnsupportedClaimCount、PlannerToolAccuracy、EvidenceGapResolutionRate。

---

## 4. 目标输出：风险初筛报告

v1 不应只返回普通 QA 文本，而应返回报告型 structured output。报告结构参考用户提供的商品合规体检报告截图，但必须保留证据引用、模型边界和初筛性质。

### 4.1 RiskScreeningReport Schema

建议定义：

```text
RiskScreeningReport
  report_id
  trace_id
  created_at
  product_profile
  target_markets
  screening_scope
  overall_verdict
  country_summaries
  risk_cards
  module_results
  evidence_items
  action_recommendations
  missing_evidence
  limitations
  langfuse_url
```

其中 `overall_verdict` 的枚举为：

```text
no_risk_found
caution
not_recommended
insufficient_evidence
```

这些枚举代表证据初筛信号，不代表最终法律结论。

### 4.2 报告页面结构

报告应包含以下章节：

1. **报告头部**
   - 报告 ID
   - 创建时间
   - 检测范围
   - 检测国家/市场
   - 商品/品牌/功能描述
   - 可选商品图片或 image asset placeholder

2. **风险统计卡片**
   - 暂未发现风险
   - 谨慎上架
   - 不建议上架
   - 证据不足

3. **检测总结**
   - 按国家输出一句结论。
   - 每条结论必须绑定证据数量、证据类型和主要风险原因。

4. **行动建议**
   - 未上架场景建议
   - 已上架场景建议
   - 无法更换商品或设计时的保守建议
   - 人工复核建议

5. **分模块详情**
   - 商标检测
   - 专利检测
   - 诉讼检测
   - 图片/外观/版权检测契约预留，v1 只定义 schema 和展示位置

6. **证据与引用**
   - evidence id
   - source type
   - matched field
   - content excerpt
   - retrieval mode
   - rank / score
   - citation

7. **缺失证据与限制**
   - 未检索到的证据类型
   - 索引不可用
   - 模型失败
   - 数据源覆盖不足

### 4.3 结论生成约束

系统不得在没有证据支持时输出“侵权”“违法”“必然下架”等结论。推荐措辞：

```text
命中高相似商标证据，初筛风险较高，建议人工复核后再决定是否上架。
```

不推荐措辞：

```text
该商品已经构成侵权。
```

如果检索失败或证据不足，应输出：

```text
overall_verdict = insufficient_evidence
```

而不是“暂未发现风险”。

---

## 5. 系统架构与模块设计

v1 采用“专业领域内核 + agentic runtime + MCP/Streamlit 外壳”的结构。

```text
Data Sources
  XML / TSV / CSV
  future PDF / HTML / Markdown / images
        |
Ingestion Layer
  parsers -> normalized documents -> chunks -> optional image asset contract
        |
Index Layer
  DuckDB exact lookup
  BM25 sparse index
  vector store / Milvus dense index
  NetworkX GraphRAG index
        |
Agentic Runtime
  query normalization
  scenario rewrite
  LLM planner
  tool-specific query rewrite
  tool execution
  evidence gap check
  bounded follow-up rewrite
  rerank
  structured report builder
        |
Interfaces
  MCP server for AI clients
  Streamlit dashboard for local inspection
  CLI for reproducible batch/eval
        |
Observability & Evaluation
  Langfuse traces
  local JSONL traces
  retrieval metrics
  RAGAS answer metrics
  regression test suite
```

### 5.1 建议模块边界

目标源码结构为：

```text
src/crossborder_agentic_rag/
  core/
    settings.py
    registry.py
    contracts.py
  ingestion/
    loaders/
    parsers/
    transforms/
  indexing/
    duckdb_index.py
    bm25_index.py
    vector_index.py
    graph_index.py
  retrieval/
    dense.py
    sparse.py
    hybrid.py
    graph.py
    rerank.py
    evidence_hit.py
  agentic/
    normalizer.py
    scenario_rewriter.py
    planner.py
    tool_query_rewriter.py
    dispatcher.py
    evidence_gap.py
    runtime.py
  reports/
    schema.py
    builder.py
    citation_audit.py
    render.py
  mcp_server/
    server.py
    tools.py
    resources.py
    schemas.py
  dashboard/
    app.py
    pages/
    services/
  observability/
    trace_schema.py
    jsonl_trace.py
    langfuse_trace.py
  evaluation/
    datasets.py
    retrieval_metrics.py
    citation_metrics.py
    agent_metrics.py
    ragas_runner.py
```

该模块结构是目标系统的职责边界。实现时应保证跨模块通信通过 schema 和 service 接口完成，避免 UI、MCP 或评测逻辑直接依赖底层脚本细节。

### 5.2 核心能力组合

目标系统由以下核心能力组合而成：

- trademark XML parser
- patent TSV parser
- litigation CSV parser
- source-aware chunking
- DuckDB exact lookup
- BM25 retrieval
- Milvus dense retrieval
- RRF fusion
- reranker baseline
- NetworkX GraphRAG
- retrieval evaluation
- RAGAS generation evaluation runner with Qwen-safe configuration

这些能力通过统一 schema 和插件接口连接，形成从数据摄取、索引构建、agentic 查询、报告生成到评测追踪的完整闭环。

---

## 6. 全链路可插拔架构

鉴于跨境电商知识产权风险初筛系统需要长期适配不同数据源、模型服务、向量数据库、评估框架与部署环境，本项目在架构上避免与单一供应商、单一数据格式或单一检索策略绑定。系统的核心原则是：

> **业务语义稳定，技术实现可替换。**

也就是说，`Trademark Evidence`、`Patent Claim Evidence`、`Litigation Evidence`、`RiskScreeningReport` 等领域对象保持稳定，而 LLM、Embedding、Reranker、Loader、Vector Store、Evaluator、Trace Backend 均通过统一接口接入。

### 6.1 LLM 调用层插拔 (LLM Provider Agnostic)

系统中的 LLM 不直接散落在 planner、rewrite、report generation、evaluation 脚本中，而是统一通过 `BaseLLMClient` 调用。该接口负责：

- 普通文本生成
- JSON 结构化输出
- provider 参数适配
- 超时与重试
- 错误归一化
- `disable_thinking=true` 等模型特定参数注入
- token usage 和 latency 记录

v1 支持或预留：

- **OpenAI-compatible API**：适配 OpenAI、Qwen、DeepSeek、vLLM、LM Studio 等兼容接口。
- **Azure OpenAI**：用于企业环境、合规部署或私有网络场景。
- **Local/Ollama/vLLM**：用于本地实验、隐私保护和低成本评测。
- **Template/Mock LLM**：用于单元测试和无 API key 的离线 smoke test。

planner、query rewrite、report synthesis、RAGAS judge 不应知道底层模型供应商，只依赖统一的 `complete_structured()` 或 `complete_text()` 能力。切换模型应只改配置，不改业务代码。

### 6.2 Embedding & Rerank 模型插拔 (Model Agnostic)

Embedding 和 Reranker 是检索质量最容易迭代的部分，因此必须独立于检索 orchestration。系统通过 `BaseEmbeddingProvider` 和 `BaseReranker` 抽象不同后端。

Embedding 可替换：

- OpenAI / Azure embedding
- BGE / E5 / GTE / Qwen embedding
- Ollama / local sentence-transformers
- Fake embedding，仅用于测试和 smoke run

Rerank 可替换：

- noop / lexical baseline
- BGE Cross-Encoder
- Cohere Rerank
- LLM rerank，作为未来扩展

系统要求所有 embedding/rerank 后端返回统一 schema，并记录模型名、维度、耗时、top-k 变化，方便 Langfuse 和 Streamlit 对比不同实验。

### 6.3 RAG Pipeline 组件插拔

当前核心数据源是 `xml / tsv / csv`，但系统不应被这三类格式锁死。v1 以 `Document`、`EvidenceChunk`、`ImageAsset` 作为稳定中间层：

```text
Raw Source
-> Loader
-> Normalized Document
-> Splitter / Chunker
-> Transform / Enrichment
-> EvidenceChunk
-> Indexes
```

可插拔组件包括：

1. **Loader（解析器）**
   - v1 支持 trademark XML、patent TSV、litigation CSV。
   - 未来支持 PDF、HTML、Markdown、图片资产。

2. **Smart Splitter（切分策略）**
   - 商标记录切分
   - 专利 claim-level 切分
   - 诉讼 docket/event 切分
   - 通用语义切分

3. **Transformation（增强逻辑）**
   - 字段标准化
   - 实体抽取
   - GraphRAG 节点抽取
   - OCR
   - Image Captioning
   - metadata enrichment

4. **Image Enrichment Contract**
   - v1 不要求现有 xml/tsv/csv 含图片。
   - 所有 Document/Chunk 都预留 `images=[]`。
   - 未来图片可通过 caption/OCR 进入同一文本检索链路。

### 6.4 检索策略插拔 (Retrieval Strategy)

系统不把 Agentic RAG 固定为一种检索方式。LLM planner 可以为不同子任务选择不同检索策略，Python runtime 负责执行和约束。

支持策略：

- `duckdb_lookup`：精确字段查询，如 patent number、case number、registration number。
- `bm25_only`：关键词和精确标识符召回。
- `dense_only`：语义相似问题、产品功能描述、专利 claim paraphrase。
- `hybrid_rrf`：BM25 + dense 的稳定融合召回。
- `hybrid_rerank`：高精度候选重排。
- `graph_rag`：实体关系扩展，如 company -> patent -> litigation case。

未来向量数据库可替换：

- Milvus
- Qdrant
- Chroma
- FAISS
- Weaviate

关键是所有检索器都返回统一 `EvidenceHit`，包含：

```text
chunk_id
source_type
content
metadata
score
rank
retrieval_mode
tool_name
citation
```

这样 report builder 不关心证据来自 DuckDB、BM25、Milvus 还是 GraphRAG。

### 6.5 评估体系插拔 (Evaluation Framework)

评估不绑定单一指标。v1 采用“本地可复现评测 + Langfuse trace”的组合：

- Retrieval metrics：HitRate@k、Recall@k、MRR、nDCG、SourceCoverage。
- Citation audit：引用是否存在、引用是否覆盖最终结论、是否出现无证据断言。
- RAGAS generation metrics：faithfulness、answer relevancy，保留 Qwen-safe `disable_thinking=true` 配置。
- Agent-step metrics：planner 工具选择是否命中预期、follow-up query 是否减少 evidence gap、每个工具耗时和失败率。
- Langfuse trace：记录 planner、rewrite、tool calls、retrieval hits、rerank、report synthesis。

未来可以接 DeepEval、TruLens 或人工标注评分，但不替换已有本地质量门。

### 6.6 可观测性与 Dashboard 插拔

Streamlit 不直接调用底层脚本，而是通过 service layer 读取统一 trace、report、evaluation artifact。Trace backend 可插拔：

- local JSONL：离线、CI、无外部服务时可用。
- Langfuse：交互式调试、模型调用追踪、评测运行对比。
- future OpenTelemetry：生产部署时扩展。

这样 UI 页面只展示统一对象，不依赖某个模型或某个检索实现。

---

## 7. MCP 生态集成

v1 将 MCP 定位为系统的主外部调用入口，但不是业务 HTTP API。MCP 主要服务 Claude Desktop、Cursor、Copilot/Codex 类 AI clients，让外部 Agent 能把本项目当作“跨境电商 IP 风险证据工具箱”来调用。

### 7.1 MCP Server 设计目标

MCP Server 不应只是把 CLI 包一层，而应暴露稳定、可组合、可追踪的工具。每次 MCP 调用都生成 `trace_id`，并能在 Streamlit/Langfuse 里复盘。

### 7.2 v1 Tools

```text
query_ip_risk
  输入：product/profile/query/target_markets/scope
  输出：RiskScreeningReport
  用途：主入口，返回报告型结果。

search_evidence
  输入：query/source_types/retrieval_mode/top_k/filters
  输出：EvidenceHit[]
  用途：给外部 AI client 精确检索证据。

lookup_structured_record
  输入：identifier/type/source_type
  输出：structured rows
  用途：查 trademark registration、patent number、case number 等精确字段。

list_sources
  输入：可选 collection/source_type
  输出：当前索引数据源、统计、更新时间
  用途：让 AI client 知道系统能查什么。

get_trace
  输入：trace_id/report_id
  输出：planner、rewrite、tool calls、retrieval/rerank/report events
  用途：审计和调试。

get_eval_report
  输入：eval_run_id 或 latest
  输出：本地评测摘要和 artifact 路径
  用途：让 AI client 判断当前系统质量状态。
```

### 7.3 MCP Resources

```text
ip-risk://reports/{report_id}
ip-risk://traces/{trace_id}
ip-risk://eval-runs/{run_id}
ip-risk://schemas/risk-screening-report
ip-risk://schemas/evidence-hit
```

### 7.4 协议输出约束

`query_ip_risk` 不能只返回自然语言，应返回：

```text
structuredContent:
  RiskScreeningReport JSON

content:
  简短中文报告摘要
  关键风险结论
  引用说明
```

如果包含图片证据，v1 先返回 image metadata 和 storage path。未来可返回 MCP `ImageContent` 或可访问 resource URI。

### 7.5 错误处理

MCP 层需要标准化错误：

```text
INVALID_INPUT
NO_INDEX_AVAILABLE
LLM_PROVIDER_UNAVAILABLE
RETRIEVER_FAILED
INSUFFICIENT_EVIDENCE
TRACE_NOT_FOUND
EVALUATION_NOT_READY
```

关键原则：失败不能伪装成“暂无风险”。如果没有索引、没有证据或模型失败，报告必须输出 `overall_verdict=insufficient_evidence`。

---

## 8. Streamlit 可视化管理平台

Streamlit 的定位不是营销页面，也不是简单聊天框，而是本地开发和演示用的风险初筛工作台。它要让人看清楚：系统为什么给出这个结论、用了哪些工具、命中了哪些证据、评测结果是否可信。

### 8.1 v1 页面

```text
1. 系统总览
2. 风险初筛报告
3. 证据浏览器
4. Query Trace 追踪
5. Ingestion / Index 管理
6. Evaluation 面板
7. 配置与插件状态
```

### 8.2 系统总览

展示当前系统配置和数据资产：

- LLM provider / model / `disable_thinking`
- embedding model / dimension
- reranker provider
- vector store backend
- DuckDB path / BM25 index / GraphRAG index
- trademark / patent / litigation chunk 数量
- 最近一次 ingestion / eval / query 时间
- Langfuse 是否启用

### 8.3 风险初筛报告

这是主页面，格式参考用户提供的报告截图：

- 报告头部：产品对象、检测范围、检测国家、创建时间、报告 ID
- 风险统计卡片：`暂未发现风险`、`谨慎上架`、`不建议上架`、`证据不足`
- 检测总结：按国家输出一句业务结论
- 行动建议：如果未上架、已上架、无法更换设计，分别给建议
- 模块详情：商标、专利、诉讼，未来图片/外观/版权
- 证据引用：每条结论下面可以展开 evidence hits

### 8.4 证据浏览器

用于检查当前索引里的证据：

- source type 过滤：trademark / patent / litigation / future image
- chunk 搜索
- metadata 展示
- citation preview
- image assets 预留展示区
- raw document / chunk_id / doc_id 反查

### 8.5 Query Trace 追踪

展示一次 query 的完整链路：

```text
query normalization
scenario rewrite
LLM planner output
tool-specific rewritten queries
tool calls
retrieval mode per tool
retrieved hits
rerank before/after
evidence gap check
follow-up rewrite
final report synthesis
```

该页面是证明系统具备 agentic workflow 的关键，而不是仅展示最终答案。

### 8.6 Ingestion / Index 管理

v1 先做轻量版：

- 查看数据源路径
- 显示已构建索引状态
- 触发 fixture/demo ingestion
- 显示 parser/index report
- 不在 v1 强行做完整大数据上传管理

### 8.7 Evaluation 面板

展示本地评测和 RAGAS 结果：

- golden query set
- retrieval metrics
- citation audit
- agent-step metrics
- RAGAS faithfulness / answer relevancy
- 历史 eval run 对比
- artifact 路径
- Langfuse trace links

### 8.8 配置与插件状态

展示每个插件是否可用：

- LLM configured / unavailable
- embedding configured / fake embedding warning
- Milvus connected / dry-run only
- Langfuse connected / local fallback
- RAGAS dependencies available

关键原则：Streamlit 页面只调用 service layer，不直接 import 一堆 scripts。这样 UI 不会成为新的混乱入口。

---

## 9. 测试与评估方案

本项目采用 **测试驱动与评估驱动并行** 的开发方式。普通软件测试验证“代码是否按接口工作”，RAG 评估验证“系统是否真的检索到正确证据、生成的报告是否可信”。两者缺一不可。

### 9.1 核心原则

- **测试即契约**：每个模块的输入输出 schema、错误行为、fallback 行为都必须有测试。
- **评估即产品质量**：风险初筛系统不能只靠 demo query 判断效果，必须用 golden queries 和离线指标衡量。
- **可复现优先**：CI 默认使用 fixture、mock LLM、fake embedding、本地 BM25/DuckDB；真实模型和 Langfuse/RAGAS 作为可选集成评估。
- **证据边界优先**：任何无法由证据支持的结论，都应被 citation audit 捕获。
- **失败显式化**：无索引、无证据、LLM 失败、RAGAS context 缺正文时，必须返回明确错误或 `insufficient_evidence`，不能误报“暂无风险”。

### 9.2 单元测试 (Unit Tests)

目标：验证每个独立组件的行为，隔离外部依赖。

| 模块 | 测试重点 | 典型测试用例 |
|---|---|---|
| Loader / Parser | XML/TSV/CSV 字段解析、缺失字段、编码、schema 输出 | USPTO trademark XML 样例、patent TSV 样例、litigation CSV 样例、坏格式输入 |
| Normalizer | 统一 Document schema、source_type、doc_id、metadata | 不同数据源归一成稳定字段 |
| ImageAsset Contract | 文本数据源默认 `images=[]`，未来图片字段不破坏既有 pipeline | Parser 默认空图片、caption/ocr 字段 schema 校验 |
| Chunker | claim-level chunk、trademark goods/services chunk、litigation event chunk | chunk_id 稳定、content 非空、metadata 继承 |
| LLM Client | provider 配置、thinking 关闭、错误归一化 | OpenAI-compatible mock、`disable_thinking=true` 注入 |
| Query Rewrite | 场景改写、工具级 query rewrite、follow-up rewrite | 用户问题改写为 trademark/patent/litigation 子查询 |
| Planner | 工具选择、retrieval mode 选择、JSON parse fallback | mock LLM 输出不同 tool_plan |
| Tool Dispatcher | 工具执行、超时、错误、fallback | DuckDB/BM25/Milvus unavailable 时受控失败 |
| Retrieval | BM25/dense/hybrid/RRF/rerank | exact ID query 走 BM25、semantic query 走 dense/hybrid |
| Evidence Evaluator | 证据缺口判断、follow-up 触发 | patent 问题缺 patent evidence 时触发 rewrite |
| Report Builder | 必填章节、verdict、country summary、action recommendations | 生成报告型 structured output |
| Citation Audit | 引用合法性、unsupported claim 检测 | 引用不存在、结论无证据时失败 |
| Trace Adapter | JSONL/Langfuse event schema | 每个阶段都有 trace event |
| MCP Schema | tool 输入输出 schema | `query_ip_risk` 返回 structuredContent |

### 9.3 集成测试 (Integration Tests)

目标：验证多个模块组合后的数据流是否正确。

| 场景 | 验证要点 | 测试策略 |
|---|---|---|
| Fixture Ingestion Pipeline | parser -> document -> chunk -> index artifact | 使用小型 xml/tsv/csv fixtures |
| Structured Lookup | normalized docs -> DuckDB -> exact lookup | registration number、patent id、case number |
| Hybrid Retrieval | BM25 + dense mock + RRF + rerank | 检查 Top-K 是否包含预期 evidence id |
| Agentic Query Flow | rewrite -> planner -> tools -> gap check -> report | mock LLM，验证 tool calls 和 trace |
| MCP Tool Call | MCP client 调用 `query_ip_risk` | 验证 JSON-RPC/MCP content 和 structuredContent |
| Streamlit Service Layer | 读取 reports/traces/eval artifacts | 不启动完整浏览器也能测试 service |
| Langfuse Fallback | Langfuse 不可用时写 local JSONL | 不丢 trace，不影响主流程 |
| RAGAS Input Builder | eval result -> RAGAS JSONL | contexts 必须包含正文 |

### 9.4 端到端测试 (End-to-End Tests)

目标：模拟真实用户使用路径，验证系统可运行。

核心 E2E 场景：

```text
场景 1：商标风险初筛
输入：商品名/品牌词/目标市场 US
预期：调用 trademark_search_tool + DuckDB/BM25/hybrid，生成报告，包含商标证据和引用。

场景 2：专利 claim 风险初筛
输入：产品功能描述
预期：调用 patent_search_tool，优先语义/hybrid 检索，报告包含 patent claim evidence。

场景 3：诉讼风险核查
输入：公司名/专利号/case number
预期：调用 litigation_search_tool 和 structured lookup，报告包含 docket/case evidence。

场景 4：混合 IP 风险初筛
输入：类似 "Can I sell a smart travel bag in US?"
预期：规划 trademark、patent、litigation 多工具，输出综合风险信号和缺失证据。

场景 5：MCP 外部调用
输入：MCP query_ip_risk
预期：返回结构化 RiskScreeningReport，并可通过 get_trace 复盘。

场景 6：图片扩展契约
输入：带 images=[] 的当前数据，或 fixture 中带 image metadata 的 mock document
预期：不破坏文本 pipeline；报告中可显示 image asset placeholder。
```

### 9.5 RAG 质量评估 (RAG Quality Evaluation)

目标：验证系统质量，而不只是代码可运行。

评估数据建议：

```text
eval/golden_queries_ip_v1.jsonl
  query
  target_market
  expected_source_types
  expected_evidence_ids
  expected_tools
  expected_retrieval_modes
  reference_answer
  reference_risk_signal
```

指标分层：

| 层级 | 指标 |
|---|---|
| Retrieval | HitRate@k, Recall@k, MRR@k, nDCG@k, SourceCoverage |
| Agentic Planning | PlannerToolAccuracy, RetrievalModeMatchRate, FollowupTriggerAccuracy |
| Evidence Gap | EvidenceGapRecall, GapResolutionRate |
| Citation | ValidCitationRate, CitationCoverage, UnsupportedClaimCount |
| Report | RequiredSectionCoverage, VerdictEvidenceAlignment, ActionRecommendationCoverage |
| RAGAS | Faithfulness, Answer/Response Relevancy, Context Precision/Recall |
| Operations | P50/P95 latency, tool failure rate, fallback rate |

关键说明：

- RAGAS 只评估生成质量，不替代 retrieval/citation 指标。
- `FaithfulnessProxy` 只能作为 heuristic，不能包装成真实事实性结论。
- 如果没有人工标注或可靠 weak labels，README/spec 不应宣称固定 Recall、nDCG、准确率。
- 对 Qwen 等 thinking 模型，评测调用必须禁用 thinking，避免 `n>1` 兼容问题。

### 9.6 性能与压力测试 (Performance Tests)

v1 是本地优先系统，不要求生产级并发，但需要建立基准：

| 测试类型 | 验证点 | 工具 |
|---|---|---|
| Query Latency | P50/P95/P99 | pytest-benchmark 或自定义 timer |
| Tool Latency Breakdown | planner/retrieval/rerank/report 各阶段耗时 | trace events |
| Index Size Scaling | chunk 数增长后的检索耗时 | fixture + synthetic data |
| RAGAS Runtime | max_workers、timeout、失败率 | saved eval artifacts |
| Memory Footprint | 大 BM25/Graph index 加载 | optional profiling |

### 9.7 CI/CD 集成

默认 CI：

```text
python -m compileall -q src scripts
pytest -q
schema contract tests
fixture E2E tests
MCP schema tests
report builder tests
citation audit tests
runtime artifact path contract tests
schema and artifact path contract tests
```

可选手动/夜间评测：

```text
real embedding retrieval eval
Milvus integration eval
RAGAS generation eval
Langfuse trace smoke test
Streamlit smoke test
```

---

## 10. 项目目录结构

修改完成后的项目目录应直接体现“基于 Agentic RAG 的跨境电商知识产权风险初筛系统”的产品形态。目录结构不以临时实验脚本为中心，而以稳定领域对象、可插拔组件、MCP 服务、Streamlit 工作台和评测闭环为中心。

### 10.1 顶层目录

```text
Agentic-RAG-CrossBorder-Marketplace/
├── README.md                         # 项目介绍、快速开始、核心能力说明
├── pyproject.toml                    # Python 包与可选依赖
├── .env.example                      # LLM、Embedding、Milvus、Langfuse 等环境变量示例
├── configs/                          # 运行配置与插件选择
├── docs/                             # DEV_SPEC、架构说明、运行手册
├── eval/                             # 小型 golden query sets 与评测模板
├── scripts/                          # 稳定 CLI 入口
├── src/crossborder_agentic_rag/       # 核心源码包
├── tests/                            # 单元、集成、E2E、评测契约测试
└── docker-compose.yml                # Milvus 等本地服务编排
```

### 10.2 源码目录

```text
src/crossborder_agentic_rag/
├── core/                             # settings、registry、通用契约
├── schemas/                          # Document、EvidenceChunk、EvidenceHit、ImageAsset、RiskScreeningReport
├── ingestion/                        # XML/TSV/CSV loader、normalizer、chunker、image contract
├── indexing/                         # DuckDB、BM25、Vector、Graph index builders
├── retrieval/                        # dense/sparse/hybrid/graph retrieval、RRF、rerank
├── agentic/                          # query rewrite、planner、dispatcher、evidence gap、runtime
├── reports/                          # report builder、citation audit、render schema
├── mcp_server/                       # MCP tools、resources、server entry
├── dashboard/                        # Streamlit app、pages、service layer
├── observability/                    # local JSONL trace、Langfuse adapter、trace schema
├── evaluation/                       # retrieval metrics、citation metrics、agent metrics、RAGAS runner
├── llm/                              # LLM / Embedding / Vision provider adapters
├── storage/                          # persistent stores and backend adapters
└── utils/                            # JSONL、logging、text utilities
```

该结构强调“每个目录对应一个稳定职责”。例如，`agentic/` 只负责工具规划式 RAG workflow，`reports/` 只负责风险初筛报告契约与生成，`mcp_server/` 只负责 MCP 协议暴露，`dashboard/` 只负责读取 service layer 后展示。

### 10.3 配置目录

```text
configs/
├── app.yaml                          # 全局运行配置
├── plugins.yaml                      # LLM、Embedding、Reranker、Retriever、Evaluator 插件选择
├── retrieval.yaml                    # top_k、candidate_k、fusion、rerank 参数
├── mcp.yaml                          # MCP tools/resources 配置
├── dashboard.yaml                    # Streamlit 页面和 artifact 路径配置
└── evaluation.yaml                   # golden set、metrics、RAGAS、Langfuse eval 配置
```

配置文件的目标是让模型、向量库、reranker、trace backend 和评测器能够通过配置切换，而不是在业务代码中硬编码。

### 10.4 CLI 入口

```text
scripts/
├── ingest.py                         # 解析 XML/TSV/CSV 并生成 normalized documents / chunks
├── build_indexes.py                  # 构建 DuckDB、BM25、Vector、Graph indexes
├── query.py                          # 单次风险初筛查询，输出 RiskScreeningReport JSON
├── evaluate.py                       # 运行本地 retrieval/citation/agent/RAGAS 评测
├── run_mcp_server.py                 # 启动 MCP Server
└── run_dashboard.py                  # 启动 Streamlit 工作台
```

这些入口对应最终产品的稳定工作流：数据准备、索引构建、风险初筛、评测、MCP 服务、可视化工作台。

### 10.5 数据与报告资产

```text
data/
├── raw/                              # 本地原始数据，不提交
├── processed/                        # normalized docs、chunks、indexes，不提交
├── images/                           # future ImageAsset storage，不提交
└── db/                               # DuckDB、BM25、Vector、Graph local stores，不提交

reports/
├── risk_screening/{report_id}/        # 风险初筛报告 JSON、HTML/export artifacts
├── traces/{trace_id}.jsonl            # local trace fallback
└── eval/{run_id}/                     # retrieval/citation/agent/RAGAS 评测结果
```

仓库只提交小型 fixtures、schema examples 和 golden query templates。真实原始数据、索引文件、模型权重、运行报告和评测输出作为本地运行资产管理。

### 10.6 文档目录

```text
docs/
├── DEV_SPEC.md                       # 本项目正式开发规格
├── ARCHITECTURE.md                   # 架构图、模块边界、数据流
├── MCP_SPEC.md                       # MCP tools/resources/schema
├── EVALUATION_SPEC.md                # 测试与评测体系
├── DASHBOARD_SPEC.md                 # Streamlit 页面与交互
└── RUNBOOK.md                        # 本地运行、索引构建、评测、故障处理
```

文档应服务于读者理解目标系统如何工作。正式 README 引导用户运行稳定入口，DEV_SPEC 解释系统设计，RUNBOOK 说明本地操作。

---

## 11. 项目排期

### Phase 0：DEV_SPEC 与工程契约

目标：形成正式 DEV_SPEC、核心 schema 草案和可执行的工程契约，为后续实现提供统一依据。

交付：

```text
docs/DEV_SPEC.md
docs/ARCHITECTURE.md
docs/MCP_SPEC.md
docs/EVALUATION_SPEC.md
核心 schema 草案
基础配置样例
```

验证：

```text
文档章节完整性检查
schema placeholder scan
配置样例字段检查
```

### Phase 1：核心 schema 与报告契约

目标：先定义稳定领域对象，不急着接 UI/MCP。

交付：

```text
Document
EvidenceChunk
EvidenceHit
ImageAsset
RiskScreeningReport
RiskVerdict
TraceEvent
EvaluationRun
```

验证：

```text
schema unit tests
report builder tests
citation audit tests
images=[] compatibility tests
```

### Phase 2：Agentic Runtime 重构

目标：把当前混在一起的 planner/runtime/dispatcher/rewrite/report 拆开。

交付：

```text
query normalizer
scenario rewriter
LLM planner
tool-specific query rewriter
tool dispatcher
evidence gap evaluator
bounded follow-up rewriter
structured report builder
thinking-disabled LLM adapter
```

验证：

```text
mock LLM planning tests
agentic flow fixture tests
tool-call trace tests
unsupported-claim tests
```

### Phase 3：可插拔配置与稳定 CLI

目标：把配置、插件和稳定命令入口固化为用户可直接运行的项目工作流。

交付：

```text
configs/app.yaml
configs/plugins.yaml
scripts/ingest.py
scripts/build_indexes.py
scripts/query.py
scripts/evaluate.py
```

验证：

```text
README commands work
CLI smoke tests
plugin factory tests
```

### Phase 4：MCP Server

目标：让外部 AI clients 能调用系统。

交付：

```text
query_ip_risk
search_evidence
lookup_structured_record
list_sources
get_trace
get_eval_report
MCP resources
```

验证：

```text
MCP schema tests
fixture MCP tool-call tests
structuredContent tests
error mapping tests
```

### Phase 5：Langfuse + Local Trace

目标：可观察每一次 agentic query。

交付：

```text
local JSONL trace adapter
Langfuse trace adapter
trace_id propagation
Streamlit trace-compatible artifact format
```

验证：

```text
Langfuse disabled fallback tests
trace event completeness tests
latency breakdown tests
```

### Phase 6：Streamlit 工作台

目标：实现报告风格的本地风险初筛页面。

交付：

```text
system overview
risk screening report
evidence browser
query trace
eval panel
plugin status
```

验证：

```text
service layer tests
Streamlit import smoke test
sample report render snapshot/manual check
```

### Phase 7：评测闭环

目标：建立统一的 retrieval、citation、agentic planning 和 RAGAS 评测闭环。

交付：

```text
golden_queries_ip_v1.jsonl
retrieval metrics runner
citation audit
agentic metrics
RAGAS qwen-safe runner
eval artifact manifest
dashboard eval viewer
```

验证：

```text
fixture eval pass
ValidCitationRate 100%
UnsupportedClaimCount 0
RAGAS input context completeness test
```

---

## 12. 可扩展性与未来展望

### 12.1 v1 工程基座

v1 必须完成：

- 单轮问题驱动的 Agentic RAG workflow
- Query rewrite、tool planning、retrieval mode selection、bounded follow-up rewrite
- trademark / patent / litigation 三类核心证据源
- DuckDB、BM25、dense/vector、GraphRAG、rerank 的可插拔访问
- RiskScreeningReport structured output
- MCP tools/resources
- Streamlit 本地工作台
- Langfuse trace + local JSONL fallback
- 本地 retrieval/citation/agent/RAGAS 评测体系
- 图片扩展契约，当前数据默认 `images=[]`

### 12.2 v1.5 / v2 Roadmap

v1.5 可考虑：

- 用户上传商品图/logo 图查询
- OCR/caption 驱动的图片相似初筛
- marketplace policy evidence loader
- PDF/HTML/Markdown loader
- 更完整的 country-specific risk profile

v2 可考虑：

- 多轮 session memory
- 多 Agent 协作
- HTTP/FastAPI 产品化接口
- OpenTelemetry
- CLIP 或多模态 embedding
- 人工审核闭环和标注平台

---

## 13. 安全边界与非目标

本系统的安全边界放在项目能力、架构和评测方案之后说明，目的是明确系统适用范围，而不是削弱项目定位。v1 的核心价值是证据驱动的知识产权风险初筛，而不是替代专业法律判断。

### 13.1 法律与业务边界

本项目不得声明自己能作出最终侵权判断、法律结论或合规许可。所有报告均应包含边界说明：

```text
本报告仅用于知识产权风险初筛和证据发现，不构成法律意见。重要商业或法律决策应由专业人士结合原始证据复核。
```

系统可以输出“命中风险信号”“建议谨慎上架”“不建议直接上架”“证据不足需复核”等初筛结论，但不能输出“已经构成侵权”“一定违法”“必须下架”等最终法律判断。

### 13.2 Agentic 能力边界

v1 是单轮问题驱动的 Agentic RAG workflow，不是完整 conversational agent。系统不提供长期记忆、用户画像、跨会话任务管理或自治执行能力。

v1 不做：

- 多轮对话记忆
- 多 Agent 协作
- 用户上传图片查询
- 生产 HTTP API
- 自动法律结论
- 完整平台政策/Temu policy QA
- CLIP 多模态向量检索
- 自动执行上架、下架、投诉或申诉动作

上述能力进入 v1.5/v2 roadmap 或长期扩展方向。

### 13.3 数据与评测边界

系统质量取决于数据覆盖、索引构建、embedding/rerank 模型、golden labels 和评测上下文完整性。没有人工标注或可靠 weak labels 时，文档和 README 不应宣称固定 Recall、nDCG、准确率或法律有效性。

---

## 14. 验收标准

v1 的验收标准：

```text
- 所有 unit + fixture integration tests 通过
- RiskScreeningReport 必填字段覆盖率 100%
- ValidCitationRate = 100% on fixture eval
- UnsupportedClaimCount = 0 on fixture eval
- MCP query_ip_risk 返回 structuredContent
- MCP get_trace 可复盘 query normalization、rewrite、planner、tool calls、retrieval、rerank、report
- Streamlit 能展示报告、证据、trace、evaluation summary、plugin status
- Langfuse unavailable 时 local trace fallback 可用
- 当前 xml/tsv/csv parser 输出 images=[] 且 schema 稳定
- Qwen/OpenAI-compatible LLM provider 默认关闭 thinking
- RAGAS input contexts 包含正文，不只有 chunk_id/source/title
- data、reports、traces、eval artifacts 的路径契约清晰，运行产物不影响源码包导入
- README 引用稳定入口，能够覆盖 ingest、build_indexes、query、evaluate、MCP、dashboard 六类工作流
- 源码包、MCP server、Streamlit service、evaluation runner 均可独立 import
```

---

## 15. 关键设计决策记录

1. **项目名称采用“系统”而不是“agent”**
   原因：v1 没有多轮对话记忆和自治任务执行。更准确表述是“基于 Agentic RAG 的跨境电商知识产权风险初筛系统”。

2. **v1 采用单 Agent 工具规划型，不采用多 Agent 协作**
   原因：单 Agent workflow 更适合风险初筛系统的可测试性、可观测性和证据链复盘；多 Agent 协作可作为后续扩展方向。

3. **v1 多模态先做图片契约，不做用户上传图片查询**
   原因：当前核心数据源是 XML/TSV/CSV。先定义 `ImageAsset` 和 `images=[]` 兼容契约，可为未来 PDF/HTML/图片数据接入打基础。

4. **MCP 优先服务 AI clients，不先做业务 HTTP API**
   原因：用户目标是让外部 AI agent clients 调用系统能力。HTTP API 可进入后续产品化 roadmap。

5. **Langfuse 做 trace，本地评测做质量门**
   原因：Langfuse 适合观察模型和工具调用链路，但可复现质量判断仍应依赖本地 golden set、retrieval metrics、citation audit 和 RAGAS artifacts。

6. **报告输出采用 structured report，而不是自由文本 QA**
   原因：风险初筛场景需要国家、范围、风险卡片、检测总结、行动建议、模块详情和证据引用，类似合规体检报告。

7. **源码包继续承载跨境 IP 风险初筛能力**
   原因：项目已有的 `crossborder_agentic_rag` 包名能够表达 Agentic RAG 技术路线；正式产品定位由 README、DEV_SPEC、RiskScreeningReport 和 MCP tools 共同呈现。
