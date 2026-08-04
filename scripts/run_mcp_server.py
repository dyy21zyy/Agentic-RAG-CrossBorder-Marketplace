import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crossborder_agentic_rag.mcp_server.server import main


if __name__ == "__main__":
    main()
