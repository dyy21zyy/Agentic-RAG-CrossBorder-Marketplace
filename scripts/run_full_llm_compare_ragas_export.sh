#!/usr/bin/env bash
set -euo pipefail

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

OUT=reports/full_llm_agentic_v2_300_ragas_export_$(date +%Y%m%d_%H%M)
mkdir -p "$OUT"

echo "$OUT" | tee /tmp/full_llm_ragas_out.txt

export LLM_PLAN_PROBE_PATH="$OUT/llm_plan_probe.jsonl"

echo "===== ENV CHECK =====" | tee "$OUT/env_check.log"
echo "LLM_PROVIDER=$LLM_PROVIDER" | tee -a "$OUT/env_check.log"
echo "LLM_MODEL=$LLM_MODEL" | tee -a "$OUT/env_check.log"
echo "OPENAI_BASE_URL=$OPENAI_BASE_URL" | tee -a "$OUT/env_check.log"
echo "OPENAI_API_KEY_SET=$([[ -n "${OPENAI_API_KEY:-}" ]] && echo yes || echo no)" | tee -a "$OUT/env_check.log"
echo "OUT=$OUT" | tee -a "$OUT/env_check.log"
echo "LLM_PLAN_PROBE_PATH=$LLM_PLAN_PROBE_PATH" | tee -a "$OUT/env_check.log"

echo "===== START FULL LLM COMPARE ====="

stdbuf -oL -eL python scripts/compare_rule_vs_agentic_online.py \
  --queries data/eval/chunk_grounded_eval_v2_multigold_300_clean.jsonl \
  --out-dir "$OUT" \
  --modes agentic_llm \
  --chunks-path "$CHUNKS_PATH" \
  --duckdb-path "$DUCKDB_PATH" \
  --use-milvus \
  --collection-name "$MILVUS_COLLECTION_NAME" \
  --embedding-provider local \
  --retrieval-mode hybrid_rerank \
  --reranker-provider local \
  --reranker-model "$RERANKER_MODEL" \
  --top-k 5 \
  --candidate-k 20 \
  --max-iterations 3 \
  --use-llm \
  --llm-provider "$LLM_PROVIDER" \
  --llm-model "$LLM_MODEL" \
  --llm-base-url "$OPENAI_BASE_URL" \
  2>&1 | tee "$OUT/run.log"

echo "===== SUMMARY ====="
cat "$OUT/summary.json" | python -m json.tool | tee "$OUT/summary.pretty.json"

echo "===== CHECK RAGAS EXPORT FIELDS ====="

python - <<'PY'
import json
from pathlib import Path

out = Path(open("/tmp/full_llm_ragas_out.txt").read().strip())
p = out / "comparison_outputs.jsonl"

rows = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]

print("file =", p)
print("n =", len(rows))
print("empty ragas_response =", sum(not r.get("ragas_response") for r in rows))
print("empty ragas_contexts =", sum(not r.get("ragas_retrieved_contexts") for r in rows))
print("avg context count =", round(sum(len(r.get("ragas_retrieved_contexts") or []) for r in rows) / len(rows), 4))

if rows:
    r = rows[0]
    print("\nfirst query:", r.get("query"))
    print("first ragas_response:", (r.get("ragas_response") or "")[:500])
    ctxs = r.get("ragas_retrieved_contexts") or []
    print("first context:", ctxs[0][:500] if ctxs else "NO_CONTEXT")
PY

echo "===== CREATE RAGAS INPUT ====="

cat > scripts/extract_ragas_input_from_comparison.py <<'PY'
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inp = Path(args.comparison)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ok = 0
    bad = 0

    with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            q = r.get("ragas_user_input") or r.get("query") or ""
            response = r.get("ragas_response") or r.get("answer_preview") or ""
            contexts = r.get("ragas_retrieved_contexts") or []
            reference = r.get("ragas_reference") or ""

            if not q or not response or not contexts:
                bad += 1
                continue

            out_row = {
                "id": r.get("id") or r.get("idx"),
                "user_input": q,
                "response": response,
                "retrieved_contexts": contexts,
                "reference": reference,

                "question": q,
                "answer": response,
                "contexts": contexts,
                "ground_truth": reference,

                "query_type": r.get("query_type"),
                "task_type": r.get("task_type"),
                "latency_ms": r.get("latency_ms"),
                "retrieval_ms": r.get("retrieval_ms"),
                "precision_at_5": r.get("precision_at_5"),
                "recall_at_5": r.get("recall_at_5"),
                "hit_at_5": r.get("hit_at_5"),
                "mrr_at_5": r.get("mrr_at_5"),
                "ndcg_at_5": r.get("ndcg_at_5"),
                "map_at_5": r.get("map_at_5"),
            }

            g.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            ok += 1

    print("comparison =", inp)
    print("out =", out)
    print("ok =", ok)
    print("bad =", bad)


if __name__ == "__main__":
    main()
PY

python -m py_compile scripts/extract_ragas_input_from_comparison.py

python scripts/extract_ragas_input_from_comparison.py \
  --comparison "$OUT/comparison_outputs.jsonl" \
  --out "$OUT/ragas_input.jsonl" \
  | tee "$OUT/extract_ragas_input.log"

echo "===== CHECK RAGAS INPUT ====="

python - <<'PY'
import json
from pathlib import Path

out = Path(open("/tmp/full_llm_ragas_out.txt").read().strip())
p = out / "ragas_input.jsonl"

rows = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]

print("file =", p)
print("n =", len(rows))
print("empty response =", sum(not r.get("response") for r in rows))
print("empty contexts =", sum(not r.get("retrieved_contexts") for r in rows))
print("avg contexts =", round(sum(len(r.get("retrieved_contexts") or []) for r in rows) / len(rows), 4))

if rows:
    r = rows[0]
    print("\nuser_input:", r["user_input"])
    print("\nresponse:", r["response"][:500])
    print("\nfirst context:", r["retrieved_contexts"][0][:500])
    print("\nreference:", r["reference"][:500])
PY

echo "===== CHECK LLM PLAN PROBE ====="

python - <<'PY'
import json
from pathlib import Path
from collections import Counter

out = Path(open("/tmp/full_llm_ragas_out.txt").read().strip())
probe = out / "llm_plan_probe.jsonl"

print("probe =", probe)
print("exists =", probe.exists())

if not probe.exists():
    raise SystemExit(0)

events = [json.loads(x) for x in open(probe, encoding="utf-8") if x.strip()]
print("event_count =", len(events))
print("events =", Counter(e.get("event") for e in events))

completed = [e for e in events if e.get("event") == "llm_plan_completed"]
print("completed =", len(completed))
print("succeeded =", sum(bool(e.get("succeeded")) for e in completed))

if completed:
    ms = [float(e.get("elapsed_ms") or 0) for e in completed]
    print("planner_ms_mean =", round(sum(ms) / len(ms), 3))
    print("first_completed =", json.dumps(completed[0], ensure_ascii=False)[:1000])
PY

echo "===== DONE ====="
echo "OUT=$OUT"
echo "RAGAS_INPUT=$OUT/ragas_input.jsonl"
