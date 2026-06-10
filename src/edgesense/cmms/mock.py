"""Mock CMMS adapter: persists work requests as JSON to a local directory.

Useful for development, demos, and per-customer pilots that haven't yet
plugged in their real CMMS adapter. Also provides an audit trail any
real CMMS adapter can read back if needed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .base import CmmsClient, WorkRequest, WorkRequestResult


class MockCmmsClient(CmmsClient):
    """Writes each work request to a JSON file under `output_dir`."""

    name = "mock"

    def __init__(self, output_dir: Path | str = "reports/work_orders") -> None:
        self.output_dir = Path(output_dir)

    def submit(self, request: WorkRequest) -> WorkRequestResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cmms_ref = f"MOCK-{request.external_id}"
        now = datetime.now(timezone.utc).isoformat()
        result = WorkRequestResult(
            request=request,
            cmms_ref=cmms_ref,
            submitted_at=now,
            cmms_url=None,
            storage_path=str(self.output_dir / f"{request.external_id}.json"),
        )
        path = self.output_dir / f"{request.external_id}.json"
        path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        return result
