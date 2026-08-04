from pathlib import Path

from crossborder_agentic_rag.config.registry import PluginRegistry
from crossborder_agentic_rag.config.settings import load_app_config


def test_load_app_config_reads_plugin_choices(tmp_path: Path):
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        "llm:\n"
        "  provider: template\n"
        "  model: template\n"
        "  disable_thinking: true\n"
        "retrieval:\n"
        "  default_mode: hybrid_rerank\n"
        "observability:\n"
        "  provider: local_jsonl\n",
        encoding="utf-8",
    )
    cfg = load_app_config(config_path)
    assert cfg.llm["disable_thinking"] is True
    assert cfg.retrieval["default_mode"] == "hybrid_rerank"


def test_plugin_registry_registers_and_builds_provider():
    registry = PluginRegistry()
    registry.register("llm", "template", lambda cfg: {"provider": cfg["provider"]})
    built = registry.build("llm", {"provider": "template"})
    assert built == {"provider": "template"}
