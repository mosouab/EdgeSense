"""Human-in-the-loop feedback: capture operator verdicts on alert episodes.

Layer 1 of the feedback system. An operator looks at an alert and labels it
either a false positive or a confirmed fault; we persist that verdict together
with an authoritative snapshot of the episode (peak score, threshold, the
contributors and diagnosis at the peak, the forecast). The store is an
append-only JSONL log per source — mirrors `cmms/mock.py`.

Later layers consume this log to adapt the model (Layer 2) and to build a
false-positive latent memory (Layer 3); the log is never mutated so those
layers can always replay it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERDICT_FALSE_POSITIVE = "false_positive"
VERDICT_CONFIRMED = "confirmed"
_VALID_VERDICTS = {VERDICT_FALSE_POSITIVE, VERDICT_CONFIRMED}


@dataclass(frozen=True)
class FeedbackRecord:
    """One operator verdict on one alert episode, with an episode snapshot."""

    feedback_id: str          # FB-<utc-compact>, also the persisted filename stem
    episode_id: str           # the alert episode this verdict refers to
    source: str               # "metropt" | "hydraulic" | "cmapss"
    verdict: str              # "false_positive" | "confirmed"
    note: str
    created_at: str           # ISO 8601 UTC

    # Authoritative episode snapshot (taken server-side, not trusted from client)
    started_at: str | None = None
    ended_at: str | None = None
    peak_score: float | None = None
    threshold: float | None = None
    contributors: list[dict] = field(default_factory=list)
    diagnosis: dict | None = None
    forecast: dict | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_feedback_id() -> str:
    return "FB-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


class FeedbackStore:
    """Append-only JSONL store, one file per source: <root>/<source>.jsonl."""

    def __init__(self, root: Path | str = "reports/feedback") -> None:
        self.root = Path(root)

    def _path(self, source: str) -> Path:
        safe = "".join(c for c in source if c.isalnum() or c in ("-", "_")) or "unknown"
        return self.root / f"{safe}.jsonl"

    def append(self, record: FeedbackRecord) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(record.source)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), default=str) + "\n")
        return path

    def list(self, source: str | None = None) -> list[dict]:
        records: list[dict] = []
        if not self.root.exists():
            return records
        files = (
            [self._path(source)] if source is not None else sorted(self.root.glob("*.jsonl"))
        )
        for path in files:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records


def build_feedback_record(
    *,
    episode_id: str,
    source: str,
    verdict: str,
    note: str,
    episode: dict | None,
) -> FeedbackRecord:
    """Assemble a FeedbackRecord from a verdict + the device's episode snapshot.

    `episode` is the dict returned by EdgeDevice.get_episode(); when it's None
    (episode already aged out) we still record the verdict with empty snapshot
    fields so the operator's judgement isn't lost.
    """

    if verdict not in _VALID_VERDICTS:
        raise ValueError(
            f"verdict must be one of {sorted(_VALID_VERDICTS)}, got {verdict!r}"
        )
    ep = episode or {}
    return FeedbackRecord(
        feedback_id=new_feedback_id(),
        episode_id=episode_id,
        source=source,
        verdict=verdict,
        note=note or "",
        created_at=datetime.now(timezone.utc).isoformat(),
        started_at=ep.get("started_at"),
        ended_at=ep.get("ended_at"),
        peak_score=ep.get("peak_score"),
        threshold=ep.get("threshold"),
        contributors=ep.get("peak_contributors") or [],
        diagnosis=ep.get("diagnosis"),
        forecast=ep.get("forecast"),
        metadata={
            "started_index": ep.get("started_index"),
            "ended_index": ep.get("ended_index"),
            "peak_index": ep.get("peak_index"),
        },
    )
