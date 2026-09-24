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
    (Hydraulic) `cycle_features` carries a (T, F) matrix and
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
    # Operator-readable description for each sensor variable. Maps the raw
    # variable name (e.g. "TP2") to a short human phrase ("Compressed air
    # pressure at compressor outlet (bar)"). Used to label the per-channel
    # attribution panel.
    feature_descriptions: dict[str, str] = field(default_factory=dict)
    # First-line maintenance prompt per channel. Surfaced under each
    # contributor row so an operator gets a concrete next step, not just a
    # sensor name to stare at.
    suggested_actions: dict[str, str] = field(default_factory=dict)
    # Multi-channel diagnostic rules. The first rule whose `requires` channels
    # are ALL present in the current top contributors is used as the root
    # cause for the diagnostic ticket. Each rule:
    #   {
    #     "name":   "Cooler degradation suspected",
    #     "requires": ["FS2", "TS1"],
    #     "action": "Inspect cooler core and fluid path ...",
    #   }
    # When no rule matches, the device falls back to a single-channel
    # diagnosis using the top contributor + its suggested_action.
    diagnosis_rules: list[dict] = field(default_factory=list)


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
            feature_descriptions={
                "TP2": "Compressor outlet pressure (bar)",
                "TP3": "Pneumatic panel pressure (bar)",
                "H1": "Cyclonic separator filter pressure (bar)",
                "DV_pressure": "Regulation valve pressure (bar)",
                "Reservoirs": "Air reservoir pressure (bar)",
                "Oil_temperature": "Compressor oil temperature (°C)",
                "Motor_current": "Motor current draw (A)",
                "COMP": "Air outlet valve state",
                "DV_eletric": "Regulation valve control signal",
                "Towers": "Active drying tower",
                "MPG": "Compressor activation trigger",
                "LPS": "Low-pressure switch (0.6 bar)",
                "Pressure_switch": "Backup pressure switch",
                "Oil_level": "Oil level low signal",
                "Caudal_impulses": "Air flow impulses",
            },
            suggested_actions={
                "TP2": "Inspect compressor outlet & verify discharge line for restrictions",
                "TP3": "Check downstream piping for leaks; inspect distribution panel",
                "H1": "Replace cyclonic separator filter and inspect housing seals",
                "DV_pressure": "Inspect regulation valve and relief mechanism; verify control signal",
                "Reservoirs": "Check storage reservoirs for leaks; inspect check valves",
                "Oil_temperature": "Verify oil level and inspect oil cooler / cooling fan",
                "Motor_current": "Inspect motor windings and bearings; check power supply phases",
                "COMP": "Test air outlet valve solenoid and control signal continuity",
                "DV_eletric": "Verify electrical control signal to regulation valve; check wiring",
                "Towers": "Inspect drying tower changeover valves and control timing",
                "MPG": "Verify compressor activation trigger threshold (8.2 bar) and circuit",
                "LPS": "Test low-pressure switch calibration and contact integrity",
                "Pressure_switch": "Cross-check redundant pressure switch operation",
                "Oil_level": "Top up oil and inspect for leak path",
                "Caudal_impulses": "Inspect flow meter and air leak path downstream",
            },
            diagnosis_rules=[
                {
                    "name": "Air-leak event suspected",
                    "requires": ["TP2", "Motor_current"],
                    "action": "Patrol distribution lines for audible leaks; inspect reservoir check valves and verify regulation valve seats.",
                },
                {
                    "name": "Regulation valve fault",
                    "requires": ["DV_pressure", "DV_eletric"],
                    "action": "Bench-test regulation valve assembly; verify control signal continuity and replace valve seal kit if leaking.",
                },
                {
                    "name": "Compressor thermal stress",
                    "requires": ["Oil_temperature", "Motor_current"],
                    "action": "Top up oil to spec, inspect oil cooler fins and verify cooling-fan operation; reduce duty cycle until checked.",
                },
                {
                    "name": "Distribution / pneumatic-panel leak",
                    "requires": ["TP2", "TP3"],
                    "action": "Walk the distribution piping and pneumatic panel with a leak detector; isolate sections to localise.",
                },
                {
                    "name": "Drying-tower switching anomaly",
                    "requires": ["Towers", "DV_eletric"],
                    "action": "Inspect drying-tower changeover valves and check timer/sequencer outputs.",
                },
            ],
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
            feature_descriptions={
                "PS1": "Cooler inlet pressure (bar)",
                "PS2": "Cooler outlet pressure (bar)",
                "PS3": "Pump outlet pressure (bar)",
                "PS4": "Valve inlet pressure (bar)",
                "PS5": "Accumulator pressure (bar)",
                "PS6": "Filter differential pressure (bar)",
                "EPS1": "Motor electrical power (W)",
                "FS1": "Flow before cooler (L/min)",
                "FS2": "Flow after cooler (L/min)",
                "TS1": "Hydraulic pump temperature (°C)",
                "TS2": "Cooler outlet temperature (°C)",
                "TS3": "Oil tank temperature (°C)",
                "TS4": "Hydraulic tank temperature (°C)",
                "VS1": "Pump vibration (mm/s)",
                "CE": "Cooling efficiency (%)",
                "CP": "Cooling power (kW)",
                "SE": "System efficiency factor (%)",
            },
            suggested_actions={
                "PS1": "Inspect cooler inlet line for blockage or restriction",
                "PS2": "Check cooler outlet path; verify cooler core flow",
                "PS3": "Inspect pump outlet for cavitation or pressure drop",
                "PS4": "Check upstream filter element and valve seat",
                "PS5": "Verify accumulator pre-charge pressure and gas tightness",
                "PS6": "Replace filter element (high ΔP indicates clogging)",
                "EPS1": "Inspect motor-pump alignment and load profile",
                "FS1": "Inspect inlet flow path for obstruction or fitting leak",
                "FS2": "Inspect cooler core and outlet line for restriction",
                "TS1": "Check pump bearings and lubrication; verify pump load",
                "TS2": "Inspect cooler core fouling; verify coolant supply",
                "TS3": "Audit oil cooling circuit and oil return line",
                "TS4": "Check overall hydraulic thermal management and fluid level",
                "VS1": "Inspect pump bearings and shaft alignment for imbalance",
                "CE": "Audit the full cooling circuit (cooler, fan, coolant flow)",
                "CP": "Verify cooling fan operation and coolant supply pressure",
                "SE": "Run full hydraulic-circuit efficiency audit",
            },
            diagnosis_rules=[
                {
                    "name": "Cooler degradation suspected",
                    "requires": ["FS2", "TS1"],
                    "action": "Inspect cooler core and fluid path; verify pump bearings and cooling-fan/coolant supply. Schedule cooler service.",
                },
                {
                    "name": "Pump mechanical wear",
                    "requires": ["VS1", "TS1"],
                    "action": "Vibration-analyse the pump; inspect bearings and shaft alignment. Plan bearing replacement at next stop.",
                },
                {
                    "name": "Filter clogging",
                    "requires": ["PS6", "PS4"],
                    "action": "Replace hydraulic filter element; inspect upstream lines for contamination source.",
                },
                {
                    "name": "Accumulator pre-charge loss",
                    "requires": ["PS5", "PS3"],
                    "action": "Verify accumulator nitrogen pre-charge and recharge to spec; inspect for gas-side leaks.",
                },
                {
                    "name": "Motor / drive overload",
                    "requires": ["EPS1", "TS1"],
                    "action": "Audit motor load and duty profile; inspect drive train alignment and lubrication.",
                },
            ],
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
                        source="uci_profile",
                        start_time=f"cycle {first}",
                        end_time=f"cycle {first}",
                        start_index=first,
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
    raise ValueError(f"Unknown data source: {name}")


def list_available_sources() -> list[dict[str, Any]]:
    return [
        {
            "name": "metropt",
            "display_name": "Metro do Porto compressor",
            "available": "true",
            "suggested_calibration": 60000,
            "natural_unit": "samples",
        },
        {
            "name": "hydraulic",
            "display_name": "UCI Hydraulic Systems (cooler fault)",
            "available": "true",
            "suggested_calibration": 200,
            "natural_unit": "cycles",
        },
    ]
