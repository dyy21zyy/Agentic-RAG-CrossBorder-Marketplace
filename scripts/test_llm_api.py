#!/usr/bin/env python
"""Smoke-test an OpenAI-compatible chat-completions API."""
from __future__ import annotations
import argparse, os, sys


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--message", default="Reply exactly: DeepSeek API OK")
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()

def main() -> int:
    args=parse_args(); key=os.getenv("OPENAI_API_KEY"); base=os.getenv("OPENAI_BASE_URL"); model=os.getenv("LLM_MODEL")
    print(f"OPENAI_API_KEY set: {bool(key)}")
    print(f"OPENAI_BASE_URL: {base or ''}")
    print(f"LLM_MODEL: {model or ''}")
    if not key:
        print("OPENAI_API_KEY is required; set it without committing secrets.", file=sys.stderr); return 2
    if not model:
        print("LLM_MODEL is required.", file=sys.stderr); return 2
    try:
        from openai import OpenAI
        client=OpenAI(api_key=key, base_url=base or None)
        resp=client.chat.completions.create(model=model,messages=[{"role":"user","content":args.message}],max_tokens=args.max_tokens,temperature=args.temperature)
    except Exception as exc:
        print(f"LLM API request failed: {exc}", file=sys.stderr); return 1
    choices=getattr(resp,"choices",None)
    if choices is None:
        print("LLM API response choices is None.", file=sys.stderr); return 1
    if not choices:
        print("LLM API response choices is empty.", file=sys.stderr); return 1
    content=getattr(getattr(choices[0],"message",None),"content",None)
    if not content:
        print("LLM API response message content is empty.", file=sys.stderr); return 1
    print(content); return 0

if __name__ == "__main__": raise SystemExit(main())
