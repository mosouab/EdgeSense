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
    """Static metadata about a dataset, used by the device + UI.

    `cycle_based` controls how the device interprets each SensorEvent:
        False -> each event is one sample. Device buffers samples and
                  scores once a sliding window of `window_length` rows
                  is available.
        True  -> each event carries a complete (T, F) `cycle_features`
                  matrix and IS one window. Device buffers cycles and
                  scores each one directly.
    """

    name: str
    display_name: str
    feature_names: list[str]
    primary_channels: list[str]
    window_length: int
    stride: int
    natural_unit: str
    suggested_calibration_units: int
    cycle_based: bool = False
    output_kind: str = "anomaly"  # "anomaly" | "anomaly+rul"


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


class HydraulicSource(DataSource):
    """UCI Hydraulic Systems: each cycle is one 60 x 17 window.

    The source picks a target component (default `cooler`) and orders cycles
    so the first half of the stream is mostly nominal (calibration data) and
    the second half mixes the remaining nominal cycles with degraded ones.
    """

    SECONDS_PER_CYCLE = 60

    def __init__(
        self,
        target_component: str = "cooler",
        seed: int = 42,
        calibration_size: int | None = None,
    ) -> None:
        self._target_component = target_component
        self._seed = seed
        self._calibration_size = calibration_size
        self._loaded = False
        self._dataset = None
        self._order: list[int] = []
        self._calib_end: int = 0
        self.spec = SourceSpec(
            name="hydraulic",
            display_name="UCI Hydraulic Systems",
            feature_names=[],
            primary_channels=["PS1", "TS1", "EPS1", "CE"],
            window_length=60,
            stride=1,
            natural_unit="cycle",
            suggested_calibration_units=200,
            cycle_based=True,
            output_kind="anomaly",
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from ..datasets.hydraulic import NOMINAL_VALUES, load_hydraulic_dataset

        self._dataset = load_hydraulic_dataset()
        nominal_value = NOMINAL_VALUES[self._target_component]
        prof = self._dataset.profile
        nominal_idx = np.where((prof[self._target_component] == nominal_value).to_numpy())[0]
        degraded_idx = np.where((prof[self._target_component] != nominal_value).to_numpy())[0]

        rng = np.random.default_rng(self._seed)
        nominal_idx = nominal_idx.copy()
        degraded_idx = degraded_idx.copy()
        rng.shuffle(nominal_idx)
        rng.shuffle(degraded_idx)

        # Calibration block sized to match the user's requested calibration count
        # so inference starts as soon as calibration finishes (no big nominal tail).
        requested = self._calibration_size or self.spec.suggested_calibration_units
        split = max(1, min(requested, len(nominal_idx)))
        calib_block = list(nominal_idx[:split])
        rest_nominal = list(nominal_idx[split:])
        rest_degraded = list(degraded_idx)

        # Interleave the rest so the operator sees a mix during inference.
        interleaved: list[int] = []
        a, b = 0, 0
        while a < len(rest_nominal) or b < len(rest_degraded):
            if a < len(rest_nominal):
                interleaved.append(int(rest_nominal[a]))
                a += 1
            if b < len(rest_degraded):
                interleaved.append(int(rest_degraded[b]))
                b += 1
        self._order = [int(i) for i in calib_block] + interleaved
        self._calib_end = len(calib_block)
        self.spec.feature_names.clear()
        self.spec.feature_names.extend(self._dataset.feature_columns)
        self._loaded = True

    async def stream(
        self,
        get_speed: Callable[[], float],
        consume_seek: Callable[[], int | None],
        stop: asyncio.Event,
        pause: asyncio.Event,
    ) -> AsyncIterator[SensorEvent]:
        from ..datasets.hydraulic import NOMINAL_VALUES

        self._ensure_loaded()
        nominal_value = NOMINAL_VALUES[self._target_component]
        n_events = len(self._order)
        feature_names = self.spec.feature_names

        idx = 0
        jumped = False
        while idx < n_events:
            if stop.is_set():
                return
            if pause.is_set():
                while pause.is_set() and not stop.is_set():
                    await asyncio.sleep(0.05)
                if stop.is_set():
                    return

            seek_target = consume_seek()
            if seek_target is not None:
                idx = max(0, min(int(seek_target), n_events - 1))
                jumped = True

            cycle_id = self._order[idx]
            cycle = self._dataset.windows[cycle_id]  # (60, 17) float32
            last_step = {
                name: float(cycle[-1, i]) for i, name in enumerate(feature_names)
            }
            profile_row = self._dataset.profile.iloc[cycle_id]
            label = int(profile_row[self._target_component] != nominal_value)
            metadata = {
                "row_index": int(idx),
                "cycle_id": int(cycle_id),
                "is_anomaly": bool(label),
                "component": self._target_component,
                "profile": {k: int(profile_row[k]) for k in profile_row.index},
            }
            if jumped:
                metadata["jumped"] = True
                jumped = False

            event_ts = datetime(2024, 1, 1) + pd.Timedelta(seconds=idx * self.SECONDS_PER_CYCLE)
            yield SensorEvent(
                timestamp=event_ts,
                index=idx,
                elapsed_simulated_seconds=idx * float(self.SECONDS_PER_CYCLE),
                features=last_step,
                cycle_features=cycle,
                metadata=metadata,
            )

            speed = max(get_speed(), 1e-6)
            await asyncio.sleep(max(self.SECONDS_PER_CYCLE / speed, 0.0))
            idx += 1

    def failure_markers(self) -> list[FailureMarker]:
        from ..datasets.hydraulic import NOMINAL_VALUES

        self._ensure_loaded()
        markers: list[FailureMarker] = []
        prof = self._dataset.profile
        component_targets = {
            "cooler": [20, 3],
            "valve": [90, 80, 73],
            "pump": [1, 2],
            "accumulator": [115, 100, 90],
        }
        marker_id = 1
        for component, severities in component_targets.items():
            for severity in severities:
                # Find the first index in the streaming order where this fault occurs.
                first = None
                for stream_idx, cycle_id in enumerate(self._order):
                    if int(prof.iloc[cycle_id][component]) == severity:
                        first = stream_idx
                        break
                if first is None:
                    continue
                # Land a handful of cycles before the fault so the operator sees the change.
                jump_index = max(0, first - 5)
                markers.append(
                    FailureMarker(
                        id=marker_id,
                        label=f"#{marker_id} {component} = {severity} (first occurrence in stream)",
                        failure_type=f"{component} fault",
                        severity=f"value {severity} (nominal = {NOMINAL_VALUES[component]})",
                        source="metropt_report",
                        start_time=f"cycle {first}",
                        end_time=f"cycle {first}",
                        start_index=first,
                        jump_index=jump_index,
                    )
                )
                marker_id += 1
        return markers


class CMAPSSSource(DataSource):
    """NASA CMAPSS FD001: streams cycle-by-cycle from chosen training units
    for calibration, then walks through test units to demonstrate inference.

    Each event is one cycle (a (1, 14) feature vector). The device needs to
    buffer 30 consecutive cycles before scoring (window_length = 30).
    """

    SECONDS_PER_CYCLE = 30

    def __init__(
        self,
        train_units_for_calibration: int = 10,
        calibration_size: int | None = None,
    ) -> None:
        self._train_units_for_calibration = train_units_for_calibration
        self._calibration_size = calibration_size
        self._loaded = False
        self._dataset = None
        self._sequence: list[dict] = []
        self._calib_end: int = 0
        self.spec = SourceSpec(
            name="cmapss",
            display_name="NASA CMAPSS turbofan",
            feature_names=[],
            primary_channels=[],
            window_length=30,
            stride=1,
            natural_unit="cycle",
            suggested_calibration_units=400,
            cycle_based=True,
            output_kind="anomaly+rul",
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from ..datasets.cmapss import MAX_RUL, load_cmapss_fd001

        self._dataset = load_cmapss_fd001()
        feature_cols = self._dataset.feature_columns
        self.spec.feature_names.clear()
        self.spec.feature_names.extend(feature_cols)
        # Heuristic primary channels: pick a few that vary visibly.
        self.spec.primary_channels[:] = feature_cols[:4]

        # Build the streaming sequence. We want exactly `requested` healthy
        # train-unit cycles in front so that calibration ends right when the
        # user expects, and inference begins on test-unit data immediately.
        requested = self._calibration_size or self.spec.suggested_calibration_units

        rng = np.random.default_rng(42)
        train_unit_ids = sorted(self._dataset.train_units.keys())
        rng.shuffle(train_unit_ids)

        sequence: list[dict] = []
        for unit_id in train_unit_ids:
            unit_df = self._dataset.train_units[unit_id]
            healthy = unit_df[unit_df["rul"] >= MAX_RUL - 1e-3]
            for _, row in healthy.iterrows():
                if len(sequence) >= requested:
                    break
                sequence.append({
                    "unit_id": int(unit_id),
                    "cycle": int(row["cycle"]),
                    "features": row[feature_cols].to_numpy(dtype=np.float32),
                    "rul": float(row["rul"]),
                    "phase": "train",
                })
            if len(sequence) >= requested:
                break
        self._calib_end = len(sequence)

        # Then a few interesting test units in order: one short, one medium, one near-failure.
        test_units = sorted(self._dataset.test_units.keys())
        chosen = []
        for target_len in (60, 150, 220):
            best = min(
                test_units,
                key=lambda uid: abs(len(self._dataset.test_units[uid]) - target_len),
            )
            if best not in chosen:
                chosen.append(best)
        for unit_id in chosen:
            unit_df = self._dataset.test_units[unit_id]
            for _, row in unit_df.iterrows():
                sequence.append({
                    "unit_id": int(unit_id),
                    "cycle": int(row["cycle"]),
                    "features": row[feature_cols].to_numpy(dtype=np.float32),
                    "rul": float(row["rul"]),
                    "phase": "test",
                })
        self._sequence = sequence
        self._loaded = True

    async def stream(
        self,
        get_speed: Callable[[], float],
        consume_seek: Callable[[], int | None],
        stop: asyncio.Event,
        pause: asyncio.Event,
    ) -> AsyncIterator[SensorEvent]:
        self._ensure_loaded()
        feature_names = self.spec.feature_names
        n_events = len(self._sequence)

        idx = 0
        jumped = False
        while idx < n_events:
            if stop.is_set():
                return
            if pause.is_set():
                while pause.is_set() and not stop.is_set():
                    await asyncio.sleep(0.05)
                if stop.is_set():
                    return

            seek_target = consume_seek()
            if seek_target is not None:
                idx = max(0, min(int(seek_target), n_events - 1))
                jumped = True

            sample = self._sequence[idx]
            features_arr = sample["features"]  # (F,)
            features = {name: float(features_arr[i]) for i, name in enumerate(feature_names)}
            metadata = {
                "row_index": int(idx),
                "unit_id": sample["unit_id"],
                "unit_cycle": sample["cycle"],
                "true_rul": sample["rul"],
                "phase_kind": sample["phase"],
            }
            if jumped:
                metadata["jumped"] = True
                jumped = False
            event_ts = datetime(2024, 1, 1) + pd.Timedelta(seconds=idx * self.SECONDS_PER_CYCLE)
            yield SensorEvent(
                timestamp=event_ts,
                index=idx,
                elapsed_simulated_seconds=idx * float(self.SECONDS_PER_CYCLE),
                features=features,
                cycle_features=features_arr.reshape(1, -1),
                metadata=metadata,
            )

            speed = max(get_speed(), 1e-6)
            await asyncio.sleep(max(self.SECONDS_PER_CYCLE / speed, 0.0))
            idx += 1

    def failure_markers(self) -> list[FailureMarker]:
        self._ensure_loaded()
        markers: list[FailureMarker] = []
        # For each test unit in the stream, find its last 30 cycles (= near-failure window).
        per_unit: dict[int, list[int]] = {}
        for stream_idx, sample in enumerate(self._sequence):
            if sample["phase"] == "test":
                per_unit.setdefault(sample["unit_id"], []).append(stream_idx)
        marker_id = 1
        for unit_id, stream_indices in per_unit.items():
            if len(stream_indices) < 30:
                continue
            # Land 30 cycles before the end so the window covers near-failure cycles.
            jump_index = stream_indices[max(0, len(stream_indices) - 30)]
            start_index = stream_indices[-1]
            last_rul = self._sequence[start_index]["rul"]
            markers.append(
                FailureMarker(
                    id=marker_id,
                    label=f"#{marker_id} unit {unit_id} — final 30 cycles (RUL ≈ {last_rul:.0f})",
                    failure_type="turbofan degradation",
                    severity=f"end-of-test RUL = {last_rul:.0f} cycles",
                    source="cmapss_test_unit",
                    start_time=f"stream idx {start_index}",
                    end_time=f"stream idx {start_index}",
                    start_index=start_index,
                    jump_index=jump_index,
                )
            )
            marker_id += 1
        return markers


def get_source(name: str, calibration_size: int | None = None) -> DataSource:
    name = name.lower()
    if name in ("metropt", "metro_pt", "metro.pt"):
        return MetroPTSource()
    if name == "hydraulic":
        return HydraulicSource(calibration_size=calibration_size)
    if name == "cmapss":
        return CMAPSSSource(calibration_size=calibration_size)
    raise ValueError(f"Unknown data source: {name}")


def list_available_sources() -> list[dict[str, Any]]:
    return [
        {
            "name": "metropt",
            "display_name": "Metro do Porto compressor",
            "available": "true",
            "output_kind": "anomaly",
            "suggested_calibration": 20000,
            "natural_unit": "samples",
        },
        {
            "name": "hydraulic",
            "display_name": "UCI Hydraulic Systems (cooler fault)",
            "available": "true",
            "output_kind": "anomaly",
            "suggested_calibration": 200,
            "natural_unit": "cycles",
        },
        {
            "name": "cmapss",
            "display_name": "NASA CMAPSS turbofan",
            "available": "true",
            "output_kind": "anomaly+rul",
            "suggested_calibration": 400,
            "natural_unit": "cycles",
        },
    ]
