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
