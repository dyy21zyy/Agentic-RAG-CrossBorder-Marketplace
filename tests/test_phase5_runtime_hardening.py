from argparse import Namespace

import pytest

from crossborder_agentic_rag.agents.llm_answer import build_evidence_context, build_grounded_answer_messages, generate_grounded_answer
from crossborder_agentic_rag.llm.chat_client import ChatResult, OpenAICompatibleChatClient, build_chat_client
from crossborder_agentic_rag.schemas.evidence import EvidenceChunk


def test_chat_client_missing_key_and_template(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False); monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c=OpenAICompatibleChatClient(None,None,"m")
    assert c.complete([{"role":"user","content":"x"}]).error == "LLM_API_KEY is not set"
    assert build_chat_client("template").complete([{"role":"user","content":"x"}]).content


def test_chat_client_openai_fallback(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False); monkeypatch.setenv("OPENAI_API_KEY","k"); monkeypatch.setenv("OPENAI_MODEL","m")
    c=build_chat_client("openai")
    assert c.api_key == "k" and c.model == "m"


def test_chat_client_missing_model():
    assert OpenAICompatibleChatClient("k",None,None).complete([{"role":"user","content":"x"}]).error == "LLM_MODEL is not set"


def test_unsupported_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        build_chat_client("bad")


def test_grounded_answer_empty_does_not_call_client():
    class C:
        def complete(self,*a,**k): raise AssertionError("called")
    out=generate_grounded_answer("q", [], C())
    assert out["llm_error"] == "No evidence available for grounded answer"


def test_evidence_context_and_prompt_policy_scope():
    e=EvidenceChunk("c1","d1","trademark","registration","Title","abcdef",{},0.5)
    ctx, manifest=build_evidence_context([e], max_chars_each=3)
    assert "[E1]" in ctx and manifest[0]["content"] == "abc"
    msgs=build_grounded_answer_messages("q", ctx, "risk", "hybrid", ["missing patent"], "agentic")
    assert "not legal advice" in msgs[0]["content"].lower()
    assert "Do not require policy evidence" in msgs[0]["content"]
    assert "pipeline_mode: agentic" in msgs[1]["content"]


def test_grounded_answer_success_and_error():
    e=EvidenceChunk("c1","d1","patent","claims","Title","content",{},0.5)
    class OK:
        def complete(self,*a,**k): return ChatResult("answer", provider="p", model="m")
    assert generate_grounded_answer("q", [e], OK())["llm_answer"] == "answer"
    class ERR:
        def complete(self,*a,**k): return ChatResult("", error="boom")
    assert generate_grounded_answer("q", [e], ERR())["llm_error"] == "boom"
