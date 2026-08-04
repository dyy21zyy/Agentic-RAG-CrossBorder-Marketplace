"""Optional MCP server entry point."""

from __future__ import annotations


def main() -> None:
    try:
        import mcp  # noqa: F401
    except ImportError:
        print("MCP server requires the optional 'mcp' package; install the mcp extra to launch it.")
        return
    print("MCP package is installed. Wire runtime dependencies before launching the v1 server.")
