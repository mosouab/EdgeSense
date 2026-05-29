"""Backward-compatibility shim.

Metro.PT-specific code now lives in `edgesense.datasets.metropt`. This module
re-exports the legacy names so existing scripts keep working.
"""

from __future__ import annotations

from .datasets.metropt import (
    MetroPTDataset,
    load_metropt_dataset,
    load_metropt_failures as load_failure_reports,
)

__all__ = [
    "MetroPTDataset",
    "load_metropt_dataset",
    "load_failure_reports",
]
