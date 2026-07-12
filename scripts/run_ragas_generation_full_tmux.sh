#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/Agentic-RAG-CrossBorder-Marketplace

set -a
source .env
set +a

OUT=$(cat /tmp/full_llm_ragas_out.txt)

unset MILVUS_URI
export EMBEDDING_MODEL=/root/autodl-tmp/models/bge-base-en-v1.5
export RAGAS_EMBEDDING_MODEL=$EMBEDDING_MODEL
export EMBEDDING_DEVICE=cuda

export OPENAI_API_KEY=${OPENAI_API_KEY}
export RAGAS_LLM_MODEL=$LLM_MODEL

echo "OUT=$OUT"
echo "RAGAS_INPUT=$OUT/ragas_input.jsonl"
echo "RAGAS_LLM_MODEL=$RAGAS_LLM_MODEL"
echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"

python scripts/run_ragas_generation_from_input.py \
  --input "$OUT/ragas_input.jsonl" \
  --out-dir "$OUT" \
  --model "$RAGAS_LLM_MODEL" \
  --base-url "$OPENAI_BASE_URL" \
  2>&1 | tee "$OUT/ragas_generation_run.log"
