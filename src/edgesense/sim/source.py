"""Data sources for the simulation.

A `DataSource` describes one asset / dataset: its sensor channels, the
natural cadence of an event, the window shape the device should use, and
an async iterator that yields events. Subclasses implement one dataset
each.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

import numpy as np
import pandas as pd

from ..datasets.metropt import load_metropt_dataset


@dataclass
class SensorEvent:
    """One unit of data emitted by a `DataSource`.

    `features` is the per-channel scalar reading for source kinds where
    each event is a single timestep (Metro.PT). For cycle-based sources
    (Hydraulic, CMAPSS) `cycle_features` carries a (T, F) matrix and
    `features` holds the last timestep so the UI can still show one
    scalar per channel.
    """

    timestamp: datetime
    index: int
    elapsed_simulated_seconds: float
    features: dict[str, float]
    cycle_features: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceSpec:
    """Static metadata about a dataset, used by the device + UI."""

    name: str
    display_name: str
    feature_names: list[str]
    primary_channels: list[str]
    window_length: int
    stride: int
    natural_unit: str
    suggested_calibration_units: int


class DataSource(ABC):
    spec: SourceSpec

    @abstractmethod
    async def stream(
        self,
        speed_multiplier: float,
        stop: asyncio.Event,
        pause: asyncio.Event,
    ) -> AsyncIterator[SensorEvent]:
        """Async iterator yielding events at simulated real-time.

        `speed_multiplier` scales the inter-event sleep. `stop` aborts.
        `pause` (when set) blocks until cleared.
        """


class MetroPTSource(DataSource):
    """Continuous Metro.PT replay. 1 event per source row, ~10 s of asset time each."""

    def __init__(self, max_rows: int | None = None) -> None:
        self._dataset = None
        self._max_rows = max_rows
        self.spec = SourceSpec(
            name="metropt",
            display_name="Metro do Porto compressor",
            feature_names=[],
            primary_channels=["TP2", "Oil_temperature", "Motor_current", "Reservoirs"],
            window_length=100,
            stride=50,
            natural_unit="sample",
            suggested_calibration_units=60_000,
        )

    def _ensure_loaded(self) -> None:
        if self._dataset is None:
            self._dataset = load_metropt_dataset()
            # Mutate the existing list in place so any references captured by
            # the device before the first event see the update.
            self.spec.feature_names.clear()
            self.spec.feature_names.extend(self._dataset.feature_columns)

    async def stream(
        self,
        speed_multiplier: float,
        stop: asyncio.Event,
        pause: asyncio.Event,
    ) -> AsyncIterator[SensorEvent]:
        self._ensure_loaded()
        df = self._dataset.data
        ts_col = self._dataset.timestamp_col
        feature_cols = self.spec.feature_names
        sampling = float(self._dataset.sampling_interval_seconds)
        if not np.isfinite(sampling) or sampling <= 0:
            sampling = 10.0
        n_rows = len(df) if self._max_rows is None else min(self._max_rows, len(df))
        sim_start = asyncio.get_event_loop().time()

        for idx in range(n_rows):
            if stop.is_set():
                return
            if pause.is_set():
                while pause.is_set() and not stop.is_set():
                    await asyncio.sleep(0.05)
                if stop.is_set():
                    return
            row = df.iloc[idx]
            event_ts = row[ts_col]
            features = {col: float(row[col]) for col in feature_cols}
            elapsed = idx * sampling
            yield SensorEvent(
                timestamp=event_ts,
                index=idx,
                elapsed_simulated_seconds=elapsed,
                features=features,
                cycle_features=None,
                metadata={"row_index": int(idx)},
            )
            # Sleep to honour the speed multiplier; one source-tick is `sampling` s of asset time.
            await asyncio.sleep(max(sampling / max(speed_multiplier, 1e-6), 0.0))


def get_source(name: str) -> DataSource:
    name = name.lower()
    if name in ("metropt", "metro_pt", "metro.pt"):
        return MetroPTSource()
    raise ValueError(f"Unknown data source: {name}")


def list_available_sources() -> list[dict[str, str]]:
    return [
        {
            "name": "metropt",
            "display_name": "Metro do Porto compressor",
            "available": "true",
        },
        {
            "name": "hydraulic",
            "display_name": "UCI Hydraulic Systems",
            "available": "false",
        },
        {
            "name": "cmapss",
            "display_name": "NASA CMAPSS turbofan",
            "available": "false",
        },
    ]
