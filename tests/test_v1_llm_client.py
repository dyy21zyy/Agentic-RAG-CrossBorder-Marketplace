import sys
import types

from crossborder_agentic_rag.llm.chat_client import (
    ChatResult,
    TemplateChatClient,
    OpenAICompatibleChatClient,
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


class _FakeMessage:
    content = '{"query_type": "trademark"}'
    reasoning_content = "private chain of thought"


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]
    usage = None

    def model_dump(self):
        return {
            "choices": [{"message": {"content": self.choices[0].message.content, "reasoning_content": self.choices[0].message.reasoning_content}}],
            "reasoning_content": self.choices[0].message.reasoning_content,
        }


def _install_fake_openai(monkeypatch, create):
    class _Completions:
        def create(self, **kwargs):
            return create(kwargs)

    class _OpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_OpenAI))


def test_openai_complete_passes_thinking_option_and_redacts_reasoning(monkeypatch):
    calls = []

    def create(kwargs):
        calls.append(kwargs)
        return _FakeResponse()

    _install_fake_openai(monkeypatch, create)
    result = OpenAICompatibleChatClient("EMPTY", "http://example.invalid/v1", "qwen-compatible").complete([])

    assert calls[0]["extra_body"] == {"enable_thinking": False}
    assert result.raw == {"choices": [{"message": {"content": '{"query_type": "trademark"}'}}]}
    assert "reasoning_content" not in str(result.raw)


def test_openai_complete_type_error_fallback_records_safe_metadata(monkeypatch):
    calls = []

    def create(kwargs):
        calls.append(kwargs)
        if "extra_body" in kwargs:
            raise TypeError("extra_body unsupported")
        return _FakeResponse()

    _install_fake_openai(monkeypatch, create)
    result = OpenAICompatibleChatClient("EMPTY", "http://example.invalid/v1", "qwen-compatible").complete([])

    assert len(calls) == 2
    assert calls[0]["extra_body"] == {"enable_thinking": False}
    assert "extra_body" not in calls[1]
    assert result.raw == {"disable_thinking_requested": True, "disable_thinking_applied": False}


def test_openai_complete_structured_parses_fenced_json(monkeypatch):
    class _FencedResponse(_FakeResponse):
        class choices:
            class _Choice:
                class message:
                    content = '```json\n{"query_type": "trademark"}\n```'
                    reasoning_content = "private chain of thought"

            _Choice = _Choice

    def create(kwargs):
        response = _FencedResponse()
        response.choices = [response.choices._Choice()]
        return response

    _install_fake_openai(monkeypatch, create)
    result = OpenAICompatibleChatClient("EMPTY", "http://example.invalid/v1", "qwen-compatible").complete_structured([], "planner")

    assert result == {"query_type": "trademark"}


def test_openai_complete_structured_parse_failure_does_not_expose_reasoning(monkeypatch):
    class _MalformedResponse(_FakeResponse):
        class choices:
            class _Choice:
                class message:
                    content = "private chain of thought\n{malformed"
                    reasoning_content = "private chain of thought"

            _Choice = _Choice

    def create(kwargs):
        response = _MalformedResponse()
        response.choices = [response.choices._Choice()]
        return response

    _install_fake_openai(monkeypatch, create)
    result = OpenAICompatibleChatClient("EMPTY", "http://example.invalid/v1", "qwen-compatible").complete_structured([], "planner")

    assert result == {
        "schema_name": "planner",
        "error": "structured_output_parse_failed",
    }
    assert "private chain of thought" not in str(result)
