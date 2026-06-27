"""Optional provider-agnostic LLM judge helpers."""
from __future__ import annotations
import json
REQUIRED={"faithfulness_score","answer_relevance_score","citation_correctness_score","completeness_score","rationale"}
def parse_judge_response(text: str) -> dict:
    try: data=json.loads(text)
    except json.JSONDecodeError as exc: return {"judge_error":f"Invalid judge JSON: {exc.msg}"}
    if not isinstance(data,dict) or not REQUIRED <= set(data): return {"judge_error":"Judge JSON missing required keys"}
    return data
def run_llm_judge(*, enabled: bool, chat_client=None, prompt: str="", **kwargs) -> dict:
    if not enabled: return {"judge_skipped": True}
    if chat_client is None: return {"judge_error":"No chat client configured"}
    try:
        text=chat_client(prompt)
        return parse_judge_response(text)
    except Exception as exc:
        return {"judge_error":str(exc)}
