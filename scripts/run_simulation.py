"""Launch the EdgeSense simulation server.

  uv run python scripts/run_simulation.py

Open http://127.0.0.1:8000 in your browser.
"""

from __future__ import annotations

from pathlib import Path
import sys

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

import uvicorn

from edgesense.sim.api import app


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
