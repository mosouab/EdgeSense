"""Warm-start showcase models: persist and load a trained, feedback-adapted
device so a demo can skip the calibrate->train wait and infer immediately.

An artifact is the dict produced by `EdgeDevice.export_state()` written with
`torch.save` (it holds tensors + numpy arrays). Files live under
`models/showcase/<source>__<name>.pt` and are git-ignored — regenerate them
with `scripts/pretrain_showcase.py`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

SHOWCASE_DIR = Path("models/showcase")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-") or "model"


def artifact_path(source: str, name: str, root: Path | str = SHOWCASE_DIR) -> Path:
    return Path(root) / f"{_safe(source)}__{_safe(name)}.pt"


def save_artifact(state: dict[str, Any], name: str, root: Path | str = SHOWCASE_DIR) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = artifact_path(state["source"], name, root)
    torch.save(state, path)
    return path


def load_artifact(source: str, name: str, root: Path | str = SHOWCASE_DIR) -> dict[str, Any]:
    path = artifact_path(source, name, root)
    if not path.exists():
        raise FileNotFoundError(f"no showcase model at {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def list_artifacts(source: str | None = None, root: Path | str = SHOWCASE_DIR) -> list[dict[str, Any]]:
    """Return lightweight summaries (no weights) of available showcase models."""

    root = Path(root)
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.pt")):
        try:
            art = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if source is not None and art.get("source") != source:
            continue
        meta = art.get("meta", {})
        # name is the part after "<source>__"
        stem = path.stem
        name = stem.split("__", 1)[1] if "__" in stem else stem
        out.append({
            "name": name,
            "source": art.get("source"),
            "threshold": art.get("threshold"),
            "adapted": meta.get("adapted", False),
            "n_patterns": meta.get("n_patterns", 0),
            "has_calibration_windows": meta.get("has_calibration_windows", False),
            "calibration_samples": meta.get("calibration_samples"),
            "created_at": meta.get("created_at"),
            "path": str(path),
        })
    return out
