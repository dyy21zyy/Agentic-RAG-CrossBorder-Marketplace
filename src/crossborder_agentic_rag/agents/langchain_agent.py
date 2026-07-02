"""LangChain single-agent builder for Phase 5 IP QA."""
from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = """You are a cross-border e-commerce intellectual property QA assistant.
The system only covers trademark, patent, and litigation evidence.
Do not use marketplace policy evidence.
Do not calculate patent expiration dates or legal deadlines.
Use only evidence returned by tools.
Do not answer from general knowledge.
If evidence is insufficient, say what evidence is missing.
When possible, cite chunk_id or doc_id from the retrieved evidence.
For mixed IP risk questions, call trademark, patent, and litigation tools unless the user clearly asks for only one source."""


def build_langchain_ip_agent(llm, tools, system_prompt: str | None = None):
    """Build a single LangChain tool-calling AgentExecutor."""
    try:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    except ImportError as exc:
        raise ImportError("LangChain agent support is required. Install with: pip install langchain langchain-core langchain-openai") from exc

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt or DEFAULT_SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, return_intermediate_steps=True, handle_parsing_errors=True)
