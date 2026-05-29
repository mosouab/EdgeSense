"""UCI Condition Monitoring of Hydraulic Systems (Helwig et al., 2015).

The dataset records 2205 cycles of a hydraulic test rig at varying sensor
rates: pressures (PS1-PS6) and motor power (EPS1) at 100 Hz, volume flows
(FS1-FS2) at 10 Hz, temperatures (TS1-TS4) plus VS1/CE/CP/SE at 1 Hz.
Each 60-second cycle has labels for cooler condition, valve condition,
internal pump leakage, accumulator pressure, and a stable-flag indicating
whether quasi-steady state was reached.

We treat each cycle as a single window. All sensors are downsampled to
1 Hz by averaging within 1-second bins, so each cycle becomes a 60 x 17
feature matrix that the USAD 1D-CNN can consume directly.

Public dataset:
https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_HYDRAULIC_DIR = Path(__file__).resolve().parents[3] / "data" / "hydraulic"

# (file_stem, sample_rate_hz)
SENSOR_FILES: list[tuple[str, int]] = [
    ("PS1", 100), ("PS2", 100), ("PS3", 100), ("PS4", 100), ("PS5", 100), ("PS6", 100),
    ("EPS1", 100),
    ("FS1", 10), ("FS2", 10),
    ("TS1", 1), ("TS2", 1), ("TS3", 1), ("TS4", 1),
    ("VS1", 1),
    ("CE", 1), ("CP", 1), ("SE", 1),
]
CYCLE_LENGTH_SECONDS = 60
PROFILE_COLUMNS = ["cooler", "valve", "pump", "accumulator", "stable"]

# Per-component nominal values. A cycle is "nominal for component X" iff
# the corresponding profile column equals this value.
NOMINAL_VALUES = {
    "cooler": 100,       # full efficiency
    "valve": 100,        # optimal switching
    "pump": 0,           # no leakage
    "accumulator": 130,  # full pressure
}


@dataclass(frozen=True)
class HydraulicDataset:
    """In-memory representation of the UCI Hydraulic Systems dataset.

    Attributes:
        windows: (num_cycles, 60, num_sensors) float32 array — one window
            per cycle, downsampled to 1 Hz.
        feature_columns: ordered sensor names matching the last axis.
        profile: per-cycle DataFrame with `cooler`, `valve`, `pump`,
            `accumulator`, `stable` columns.
    """

    windows: np.ndarray
    feature_columns: list[str]
    profile: pd.DataFrame


def load_hydraulic_dataset(data_dir: Path = DEFAULT_HYDRAULIC_DIR) -> HydraulicDataset:
    """Load all 17 sensors plus the per-cycle label profile."""

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Hydraulic data dir {data_dir} not found. "
            "Run: curl -fL https://archive.ics.uci.edu/static/public/447/"
            "condition+monitoring+of+hydraulic+systems.zip -o data/hydraulic.zip "
            "&& unzip data/hydraulic.zip -d data/hydraulic"
        )

    sensor_arrays: list[np.ndarray] = []
    feature_columns: list[str] = []
    for stem, rate in SENSOR_FILES:
        path = data_dir / f"{stem}.txt"
        raw = np.loadtxt(path, dtype=np.float32)
        if raw.ndim != 2:
            raise ValueError(f"{path} did not load as 2D array (got {raw.shape}).")
        expected_cols = CYCLE_LENGTH_SECONDS * rate
        if raw.shape[1] != expected_cols:
            raise ValueError(
                f"{path} has {raw.shape[1]} columns; expected {expected_cols} "
                f"(rate {rate} Hz x {CYCLE_LENGTH_SECONDS} s)."
            )
        # Downsample to 1 Hz by averaging within each 1-second bin.
        downsampled = raw.reshape(raw.shape[0], CYCLE_LENGTH_SECONDS, rate).mean(axis=2)
        sensor_arrays.append(downsampled)
        feature_columns.append(stem)

    # Stack into (num_cycles, 60, num_sensors).
    windows = np.stack(sensor_arrays, axis=-1).astype(np.float32)

    profile_path = data_dir / "profile.txt"
    profile = pd.read_csv(profile_path, sep="\t", header=None, names=PROFILE_COLUMNS)
    if len(profile) != windows.shape[0]:
        raise ValueError(
            f"profile.txt has {len(profile)} rows; sensors have {windows.shape[0]} cycles."
        )

    return HydraulicDataset(
        windows=windows,
        feature_columns=feature_columns,
        profile=profile,
    )


def component_split(
    dataset: HydraulicDataset, component: str, train_fraction: float = 0.7, seed: int = 42
) -> dict:
    """Split cycles for per-component fault detection.

    Training set: cycles where `component` is at its nominal value (one of
    the per-component NOMINAL_VALUES). Within those, take `train_fraction`
    at random for training and the rest for the post-training healthy
    threshold calibration.

    Test set: all cycles, with binary label `1` for `component != nominal`.
    """

    if component not in NOMINAL_VALUES:
        raise ValueError(f"Unknown component {component!r}.")

    nominal_value = NOMINAL_VALUES[component]
    nominal_mask = (dataset.profile[component] == nominal_value).to_numpy()
    nominal_idx = np.where(nominal_mask)[0]
    if nominal_idx.size < 50:
        raise ValueError(
            f"Component {component} has only {nominal_idx.size} nominal cycles."
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(nominal_idx)
    n_train = int(len(nominal_idx) * train_fraction)
    train_idx = np.sort(nominal_idx[:n_train])
    calib_idx = np.sort(nominal_idx[n_train:])

    # Test set: every cycle. Labels are positive when the component is faulty.
    test_idx = np.arange(len(dataset.profile))
    test_labels = (dataset.profile[component] != nominal_value).to_numpy().astype(np.int8)

    return {
        "component": component,
        "nominal_value": nominal_value,
        "train_idx": train_idx,
        "calib_idx": calib_idx,
        "test_idx": test_idx,
        "test_labels": test_labels,
    }
