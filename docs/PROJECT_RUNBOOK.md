# Agentic-RAG-CrossBorder-Marketplace 项目运行手册

## 1. 项目简介

本项目面向跨境电商场景下的知识产权风险问答，目标是构建一个可调用多类检索工具的 Agentic RAG 系统。

系统支持的问题类型包括：

```text
1. 商标检索与商品类别分析
2. 专利 claim 检索与解释
3. 专利诉讼案件检索
4. 跨来源 IP 风险初筛
5. 实体关系扩展与 GraphRAG 检索
```

整体链路：

```text
User Query
-> LLM Planner
-> Tool Routing
-> DuckDB / BM25 / Dense Vector Search / Hybrid Rerank / GraphRAG
-> Evidence Selection
-> LLM Answer Generation
-> Evaluation
```

---

## 2. 原始数据源说明

本项目的数据主要来自公开知识产权数据和平台政策数据。原始数据通常不上传 GitHub，只在本地或服务器环境中保存。

### 2.1 商标数据源

商标数据来自 USPTO Trademark Full Text XML Data。

使用的数据类型：

```text
Trademark Full Text XML Data (No Images) – Annual Applications
Product Identifier: TRTYRAP
```

主要用途：

```text
1. 提取商标名称 word mark
2. 提取 serial number / registration number
3. 提取 Nice class
4. 提取 goods/services 描述
5. 构建 trademark evidence chunks
6. 支持商标类问答和跨境电商侵权风险初筛
```

典型字段：

```text
word_mark
serial_number
registration_number
nice_class
goods_services
owner
filing_date
status
```

如果后续使用新的商标年度数据，需要重新执行：

```text
XML parsing
-> field normalization
-> evidence chunk generation
-> BM25 / vector / structured index rebuild
```

### 2.2 专利数据源

专利数据来自 PatentsView Granted Patent Long Text Data。

使用的数据类型：

```text
PatentsView Granted Patent Long Text Data
```

主要用途：

```text
1. 提取 granted patent 信息
2. 提取 patent claims
3. 按 patent_id 和 claim_number 构建 claim-level chunks
4. 支持 patent claim 检索、claim 解释和同专利 supporting claims 检索
```

典型字段：

```text
patent_id
claim_number
claim_text
claim_sequence
claim_dependency
```

当前项目中 patent claim chunk 的典型格式：

```json
{
  "chunk_id": "patent:12186432:patent_claim:claim-1-1",
  "doc_id": "patent:12186432",
  "source_type": "patent",
  "source_subtype": "patent_claim",
  "title": "Patent 12186432 claim 1",
  "content": "Patent 12186432 Claim 1: ...",
  "metadata": {
    "claim_number": "1",
    "patent_id": "12186432"
  }
}
```

如果后续使用新的 patent long text 数据，需要保持：

```text
chunk_id
doc_id
patent_id
claim_number
source_type
source_subtype
content
metadata
```

这些字段稳定，这样评估集和引用逻辑可以继续复用。

### 2.3 专利诉讼数据源

专利诉讼数据来自 Patent Litigation Docket Reports Data。

常见文件包括：

```text
cases.csv
documents.csv
names.csv
patents.csv
```

主要用途：

```text
1. 构建 litigation case evidence
2. 连接 case number、party、patent、document、docket 信息
3. 支持 litigation overview、case timeline、party-patent relationship 查询
4. 支持 GraphRAG 中的实体关系扩展
```

典型字段：

```text
case_number
doc_number
party_name
patent_id
document_description
filing_date
court
```

这些数据通常会同时进入：

```text
1. evidence chunks
2. DuckDB structured tables
3. GraphRAG entity graph
```

### 2.4 平台政策数据源

平台政策数据来自跨境电商平台的 IP policy / marketplace policy 文档。

当前项目主要用于：

```text
1. 补充电商平台侧 IP 风险规则
2. 支持 trademark / patent / litigation evidence 与平台合规风险的结合分析
3. 支持卖家视角的侵权风险初筛回答
```

典型内容包括：

```text
intellectual property policy
trademark infringement
patent infringement
counterfeit goods
listing removal
seller compliance
```

---

## 3. 数据处理主线

原始数据不会直接进入 RAG 系统，而是经过以下处理链路：

```text
Raw Data
-> Parsing
-> Cleaning
-> Normalization
-> Chunking
-> Evidence JSONL
-> Embedding
-> Vector Database
-> BM25 Index
-> DuckDB Structured Tables
-> Graph Index
```

当前核心 evidence chunk 文件：

```text
data/processed/ip_evidence_chunks_full_optimized_fixed.jsonl
```

该文件是当前项目后续重建索引的基础。

后续如果更换：

```text
1. embedding 模型
2. reranker 模型
3. 向量数据库
4. BM25 实现
5. GraphRAG 构建方式
```

通常可以从该 JSONL 文件重新构建索引，不一定需要重新解析全部原始数据。

---

## 4. 当前主要本地文件

本地运行时需要准备以下文件：

```text
data/processed/ip_evidence_chunks_full_optimized_fixed.jsonl
data/processed/ip_structured.duckdb
data/processed/milvus_lite/ip_rag_milvus.db
data/processed/graph_index_full/ip_graph.pkl
```

文件说明：

```text
ip_evidence_chunks_full_optimized_fixed.jsonl
- 统一的 evidence chunk 文件
- 包含 trademark / patent / litigation / policy 等证据
- 后续重建向量索引、BM25 索引、RAGAS contexts 时都会用到

ip_structured.duckdb
- 结构化查询数据库
- 用于精确查询 patent_id、case_number、word_mark、class 等字段

milvus_lite/ip_rag_milvus.db
- Milvus Lite 向量库
- 存储 chunk embedding
- 用于 dense retrieval

graph_index_full/ip_graph.pkl
- NetworkX GraphRAG 图索引
- 用于实体关系扩展
```

这些文件一般不提交到 GitHub。

---

## 5. 本地模型路径

当前项目使用本地 embedding 和 reranker 模型。

```text
/root/autodl-tmp/models/bge-base-en-v1.5
/root/autodl-tmp/models/bge-reranker-base
```

用途：

```text
bge-base-en-v1.5
- 用于生成 dense embedding

bge-reranker-base
- 用于对候选 evidence 进行 rerank
```

如果后续更换 embedding 模型，例如：

```text
bge-large-en
e5-base
e5-large
gte
Qwen embedding
OpenAI embedding
```

需要重新执行：

```text
1. 读取 ip_evidence_chunks_full_optimized_fixed.jsonl
2. 用新 embedding 模型重新编码
3. 重建 Milvus / Chroma / FAISS / Qdrant / 其他向量库
4. 保持 chunk_id 不变
5. 重新跑 retrieval evaluation
```

---

## 6. 环境变量配置

每次运行前建议执行：

```bash
cd /root/autodl-tmp/Agentic-RAG-CrossBorder-Marketplace

set -a
source .env
set +a

unset MILVUS_URI

export RAG_MILVUS_URI=/root/autodl-tmp/Agentic-RAG-CrossBorder-Marketplace/data/processed/milvus_lite/ip_rag_milvus.db
export MILVUS_COLLECTION_NAME=ip_rag_collection

export EMBEDDING_PROVIDER=local
export EMBEDDING_MODEL=/root/autodl-tmp/models/bge-base-en-v1.5
export RAG_EMBEDDING_MODEL=$EMBEDDING_MODEL
export LOCAL_EMBEDDING_MODEL=$EMBEDDING_MODEL

export RERANKER_PROVIDER=local
export RERANKER_MODEL=/root/autodl-tmp/models/bge-reranker-base

export CHUNKS_PATH=data/processed/ip_evidence_chunks_full_optimized_fixed.jsonl
export DUCKDB_PATH=data/processed/ip_structured.duckdb
export RAG_GRAPH_PATH=data/processed/graph_index_full/ip_graph.pkl

export EMBEDDING_DEVICE=cuda
export RERANKER_DEVICE=cuda
export CUDA_VISIBLE_DEVICES=0
```

LLM 相关配置写在 `.env` 中，例如：

```bash
LLM_PROVIDER=openai
LLM_MODEL=qwen3.7-plus
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
OPENAI_API_KEY=your_api_key_here
```

注意：

```text
.env 不要提交到 GitHub。
```

---

## 7. 单条 Agentic RAG 运行

示例：

```bash
python scripts/run_agentic_rag.py \
  --query "Retrieve trademark profile evidence for MUCHO, including identity, Nice class, and goods/services." \
  --use-llm \
  --output-json \
  --show-trace \
  --show-sources
```

重点检查：

```text
1. llm_answer 是否非空
2. llm_error 是否为 None
3. tool_calls 是否存在
4. retrieved_evidence / reranked_evidence 是否存在
5. trace 中是否有 tool planning 和 tool execution
```

---

## 8. LLM Planner 验证

如果需要验证 LLM planner 是否真的被调用，可以设置：

```bash
export LLM_PLAN_PROBE_PATH=/tmp/llm_plan_probe.jsonl
```

然后运行：

```bash
python scripts/run_agentic_rag.py \
  --query "Retrieve trademark profile evidence for MUCHO, including identity, Nice class, and goods/services." \
  --use-llm \
  --output-json \
  --show-trace \
  --show-sources \
  > /tmp/verify_full_llm_with_plan_probe.json
```

检查 probe：

```bash
cat /tmp/llm_plan_probe.jsonl
```

需要确认：

```text
llm_plan_attempted = true
llm_plan_completed = true
tool_names 非空
```

---

## 9. 批量 Agentic RAG 运行

全量运行建议放在 tmux 中：

```bash
tmux new -d -s full_llm_ragas "bash scripts/run_full_llm_compare_ragas_export.sh"
```

查看输出目录：

```bash
cat /tmp/full_llm_ragas_out.txt
```

查看日志：

```bash
OUT=$(cat /tmp/full_llm_ragas_out.txt)
tail -f "$OUT/run.log"
```

主要输出文件：

```text
comparison_outputs.jsonl
ragas_input.jsonl
summary.json
```

---

## 10. RAGAS 输入注意事项

RAGAS 需要：

```text
user_input
response
retrieved_contexts
reference
```

注意：`retrieved_contexts` 必须包含 evidence 正文，不能只有：

```text
chunk_id
source
title
```

否则 faithfulness 会被严重低估。

如果 `ragas_input.jsonl` 中 contexts 只有 chunk_id / source / title，需要用 chunk 文件补全文：

```bash
OUT=$(cat /tmp/full_llm_ragas_out.txt)

python scripts/enrich_ragas_contexts_from_chunks.py \
  --input "$OUT/ragas_input.jsonl" \
  --chunks data/processed/ip_evidence_chunks_full_optimized_fixed.jsonl \
  --output "$OUT/ragas_input_contextfull.jsonl" \
  --max-context-chars 1800
```

---

## 11. RAGAS 评估

qwen3.7-plus 在 thinking 模式下可能要求 `n=1`，而 RAGAS 的 answer_relevancy 默认可能请求多个 generations。

因此建议使用 qwen-safe RAGAS 脚本：

```bash
python scripts/run_ragas_generation_qwen_safe.py \
  --input "$OUT/ragas_input_contextfull.jsonl" \
  --out-dir "$OUT/ragas_contextfull_full" \
  --metrics faithfulness,answer_relevancy \
  --model "$RAGAS_LLM_MODEL" \
  --base-url "$OPENAI_BASE_URL" \
  --timeout 300 \
  --max-workers 2 \
  --max-retries 2
```

其中 qwen-safe 脚本需要保证：

```text
1. disable_thinking = True
2. n = 1
3. answer_relevancy strictness = 1
4. timeout 足够大
5. max_workers 不要过高
```

---

## 12. 更换向量模型或向量数据库

后续如果要更换 embedding 模型或向量数据库，不需要改 Agentic RAG 主逻辑，主要重建索引即可。

### 12.1 更换 embedding 模型

流程：

```text
1. 保持 chunk 文件不变
2. 修改 EMBEDDING_MODEL
3. 重新对 chunks 编码
4. 重建向量库
5. 保持 chunk_id / doc_id / source_type / source_subtype 不变
6. 重新跑 retrieval evaluation
```

建议保持：

```text
chunk_id 不变
doc_id 不变
metadata 不变
```

这样可以复用原有评估集。

### 12.2 更换向量数据库

当前使用 Milvus Lite。后续可以替换为：

```text
Milvus Server
Chroma
FAISS
Qdrant
Weaviate
Elasticsearch dense vector
```

替换流程：

```text
1. 从 ip_evidence_chunks_full_optimized_fixed.jsonl 读取 chunks
2. 用指定 embedding 模型生成向量
3. 写入新的 vector database
4. 在 retriever 层适配 search 接口
5. 保证返回结果中包含 chunk_id、content、metadata
```

### 12.3 更换 BM25 / hybrid 检索

如果更换 BM25 实现或 hybrid 策略，需要保证：

```text
1. BM25 返回 chunk_id
2. Dense 返回 chunk_id
3. RRF / reranker 后仍保留 chunk_id
4. final evidence 中保留 content
5. RAGAS contexts 能拿到正文
```

---

## 13. GitHub 提交注意事项

不要提交：

```text
.env
API key
模型权重
原始大数据
Milvus DB
DuckDB DB
GraphRAG pkl
reports
logs
```

建议提交：

```text
src/
scripts/
configs/
docs/
README.md
requirements.txt 或 pyproject.toml
.env.example
.gitignore
少量 sample data
```

---

## 14. 常见问题

### 14.1 Milvus URI 配置问题

如果同时设置了 `MILVUS_URI` 和 `RAG_MILVUS_URI`，可能会导致连接错误。

建议运行前执行：

```bash
unset MILVUS_URI
export RAG_MILVUS_URI=/root/autodl-tmp/Agentic-RAG-CrossBorder-Marketplace/data/processed/milvus_lite/ip_rag_milvus.db
```

### 14.2 RAGAS faithfulness 异常偏低

先检查 `retrieved_contexts` 是否包含正文：

```bash
python - <<'PY'
import json
from pathlib import Path

out = Path(open("/tmp/full_llm_ragas_out.txt").read().strip())
p = out / "ragas_input.jsonl"

rows = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
r = rows[0]

print(r["user_input"])
for i, c in enumerate(r["retrieved_contexts"], 1):
    print("\n--- context", i, "len =", len(c), "---")
    print(c[:1000])
PY
```

如果只有 chunk_id / source / title，需要生成 `ragas_input_contextfull.jsonl`。

### 14.3 Qwen 模型 RAGAS 报 n 参数错误

如果出现：

```text
The n parameter must be 1 when enable_thinking is true
```

需要：

```text
1. 禁用 thinking
2. 设置 n=1
3. answer_relevancy strictness=1
```

### 14.4 tmux session 立刻消失

通常是命令行引号错误或脚本参数错误。

建议不要直接在 `tmux new` 中写复杂多行命令，而是先写 shell 脚本，再用 tmux 调用：

```bash
tmux new -d -s task_name "bash scripts/run_xxx.sh"
```

