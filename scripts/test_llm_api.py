#!/usr/bin/env python
"""Smoke-test a provider-neutral OpenAI-compatible chat-completions API."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/"src") not in sys.path: sys.path.insert(0,str(ROOT/"src"))
from crossborder_agentic_rag.llm.chat_client import build_chat_client

def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llm-provider", default=os.getenv("LLM_PROVIDER","openai_compatible")); p.add_argument("--llm-model", default=os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL")); p.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")); p.add_argument("--message", default="Reply with a short OK message."); p.add_argument("--max-tokens", type=int, default=50); p.add_argument("--temperature", type=float, default=0.0); return p.parse_args()

def main() -> int:
    args=parse_args(); key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    print(f"provider: {args.llm_provider}"); print(f"base_url: {args.llm_base_url or ''}"); print(f"model: {args.llm_model or ''}"); print(f"api_key_set: {bool(key)}")
    try:
        client=build_chat_client(provider=args.llm_provider, base_url=args.llm_base_url, model=args.llm_model, default_max_tokens=args.max_tokens, default_temperature=args.temperature)
        res=client.complete([{"role":"user","content":args.message}], temperature=args.temperature, max_tokens=args.max_tokens)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr); return 2
    if res.error:
        print(f"error: {res.error}", file=sys.stderr);
        if res.error == "LLM_API_KEY is not set": print("OPENAI_API_KEY is required; set it without committing secrets.", file=sys.stderr)
        return 1
    print(res.content); return 0
if __name__ == "__main__": raise SystemExit(main())
