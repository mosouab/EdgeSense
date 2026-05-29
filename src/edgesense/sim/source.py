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
from typing import Any, AsyncIterator, Callable

import numpy as np
import pandas as pd

from ..datasets.metropt import load_metropt_dataset, load_metropt_failures


@dataclass
class SensorEvent:
    """One unit of data emitted by a `DataSource`.

    `features` is the per-channel scalar reading for source kinds where
    each event is a single timestep (Metro.PT). For cycle-based sources
    (Hydraulic, CMAPSS) `cycle_features` carries a (T, F) matrix and
    `features` holds the last timestep so the UI can still show one
    scalar per channel.

    `metadata["jumped"] = True` is set on the first event after a seek so
    downstream consumers can reset any rolling state that depended on
    contiguous history.
    """

    timestamp: datetime
    index: int
    elapsed_simulated_seconds: float
    features: dict[str, float]
    cycle_features: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureMarker:
    """A point of interest in the source that the UI can jump to."""

    id: int
    label: str
    failure_type: str
    severity: str
    source: str
    start_time: str
    end_time: str
    start_index: int
    jump_index: int


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
        get_speed: Callable[[], float],
        consume_seek: Callable[[], int | None],
        stop: asyncio.Event,
        pause: asyncio.Event,
    ) -> AsyncIterator[SensorEvent]:
        """Async iterator yielding events at simulated real-time.

        `get_speed()` is called each iteration so the multiplier can be
        adjusted live without restarting the stream. `consume_seek()` is
        called each iteration; if it returns a non-None row index the
        source fast-forwards to that index and tags the next event with
        `metadata["jumped"] = True`. `stop` aborts. `pause` (when set)
        blocks until cleared.
        """

    def failure_markers(self) -> list[FailureMarker]:
        """Override per-source to expose jump targets. Default: none."""

        return []


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
        get_speed: Callable[[], float],
        consume_seek: Callable[[], int | None],
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

        idx = 0
        jumped = False
        while idx < n_rows:
            if stop.is_set():
                return
            if pause.is_set():
                while pause.is_set() and not stop.is_set():
                    await asyncio.sleep(0.05)
                if stop.is_set():
                    return

            seek_target = consume_seek()
            if seek_target is not None:
                idx = max(0, min(int(seek_target), n_rows - 1))
                jumped = True

            row = df.iloc[idx]
            event_ts = row[ts_col]
            features = {col: float(row[col]) for col in feature_cols}
            elapsed = idx * sampling
            metadata: dict[str, Any] = {"row_index": int(idx)}
            if jumped:
                metadata["jumped"] = True
                jumped = False
            yield SensorEvent(
                timestamp=event_ts,
                index=idx,
                elapsed_simulated_seconds=elapsed,
                features=features,
                cycle_features=None,
                metadata=metadata,
            )
            # Read speed fresh each tick so /speed updates take effect live.
            speed = max(get_speed(), 1e-6)
            await asyncio.sleep(max(sampling / speed, 0.0))
            idx += 1

    def failure_markers(self) -> list[FailureMarker]:
        """Map each known Metro.PT failure to a row index we can seek to.

        `jump_index` lands ~10 minutes of asset-time before the failure
        starts so the operator can watch the score climb from baseline.
        """

        self._ensure_loaded()
        df = self._dataset.data
        ts_col = self._dataset.timestamp_col
        sampling = float(self._dataset.sampling_interval_seconds)
        if not np.isfinite(sampling) or sampling <= 0:
            sampling = 10.0
        lead_rows = int(round(600.0 / sampling))  # ~10 minutes

        failures = load_metropt_failures()
        timestamps = pd.to_datetime(df[ts_col], errors="raise").to_numpy()
        markers: list[FailureMarker] = []
        for _, row in failures.iterrows():
            start_ts = pd.to_datetime(row["start_time"])
            # First row whose timestamp >= start_ts.
            start_index = int(np.searchsorted(timestamps, np.datetime64(start_ts)))
            if start_index >= len(df):
                continue
            jump_index = max(0, start_index - lead_rows)
            markers.append(
                FailureMarker(
                    id=int(row["failure_id"]),
                    label=f"#{int(row['failure_id'])} {row['failure_type']} — {start_ts.strftime('%b %d, %Y %H:%M')}",
                    failure_type=str(row["failure_type"]),
                    severity=str(row["severity"]),
                    source=str(row["source"]),
                    start_time=str(start_ts),
                    end_time=str(pd.to_datetime(row["end_time"])),
                    start_index=start_index,
                    jump_index=jump_index,
                )
            )
        return markers


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
