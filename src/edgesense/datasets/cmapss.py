"""NASA CMAPSS turbofan engine degradation dataset.

We use subset FD001 (100 train units, 100 test units, one operating
condition, one fault mode). Each row of train_FD001.txt is one cycle of
one engine; engines run from cycle 1 to failure in the training set, and
for some number of cycles before truncation in the test set. RUL_FD001.txt
gives the ground-truth remaining cycles at the end of each test sequence.

Convention used here:
- piecewise-linear RUL target clipped at MAX_RUL=125 (standard practice
  in the CMAPSS literature),
- 14 informative sensors after dropping constants in FD001,
- sliding window of WINDOW_LENGTH cycles, stride 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CMAPSS_DIR = Path(__file__).resolve().parents[3] / "data" / "cmapss"
WINDOW_LENGTH = 30
MAX_RUL = 125  # piecewise-linear ceiling on the RUL target

# Columns in raw CMAPSS .txt: unit, cycle, 3 operational settings, 21 sensors.
RAW_COLUMNS = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors that are constant or near-constant in FD001 and contribute no
# information; dropped following Heimes (2008) and many CMAPSS papers.
FD001_DROP_SENSORS = [
    "sensor_1", "sensor_5", "sensor_6", "sensor_10", "sensor_16", "sensor_18", "sensor_19",
]
FD001_FEATURE_COLUMNS = [c for c in RAW_COLUMNS if c.startswith("sensor_") and c not in FD001_DROP_SENSORS]


@dataclass(frozen=True)
class CMAPSSDataset:
    """Parsed FD001 split: per-unit cycle-level sensor readings + RUL targets.

    Attributes:
        train_units: dict mapping unit_id -> DataFrame indexed by cycle
            containing the feature columns plus a `rul` column with the
            piecewise-linear clipped target.
        test_units: same shape as train_units; `rul` is filled forward
            using the ground-truth end-of-sequence RUL plus the remaining
            cycle distance.
        feature_columns: ordered list of sensor columns used as input.
    """

    train_units: dict[int, pd.DataFrame]
    test_units: dict[int, pd.DataFrame]
    feature_columns: list[str]


def load_cmapss_fd001(data_dir: Path = DEFAULT_CMAPSS_DIR) -> CMAPSSDataset:
    """Load and prepare CMAPSS FD001."""

    if not (data_dir / "train_FD001.txt").exists():
        raise FileNotFoundError(
            f"CMAPSS FD001 not found in {data_dir}. Expected train_FD001.txt,"
            " test_FD001.txt, RUL_FD001.txt."
        )

    train_df = pd.read_csv(
        data_dir / "train_FD001.txt", sep=r"\s+", header=None, names=RAW_COLUMNS
    )
    test_df = pd.read_csv(
        data_dir / "test_FD001.txt", sep=r"\s+", header=None, names=RAW_COLUMNS
    )
    rul_df = pd.read_csv(
        data_dir / "RUL_FD001.txt", sep=r"\s+", header=None, names=["rul"]
    )

    feature_columns = FD001_FEATURE_COLUMNS

    # Train RUL = (max_cycle for that unit) - (current cycle), clipped at MAX_RUL.
    max_cycle_per_unit = train_df.groupby("unit")["cycle"].transform("max")
    train_df = train_df.copy()
    train_df["rul"] = np.minimum(max_cycle_per_unit - train_df["cycle"], MAX_RUL)

    # Test RUL: each unit's last-cycle RUL is given by RUL_FD001; earlier
    # cycles get last_RUL + (last_cycle - cycle), still clipped.
    test_df = test_df.copy()
    test_units_rul = {}
    for unit_id, group in test_df.groupby("unit"):
        last_cycle = int(group["cycle"].max())
        last_rul = float(rul_df.iloc[unit_id - 1, 0])
        test_df.loc[group.index, "rul"] = np.minimum(
            last_rul + (last_cycle - group["cycle"]), MAX_RUL
        )
        test_units_rul[unit_id] = last_rul

    train_units = {int(uid): g.reset_index(drop=True) for uid, g in train_df.groupby("unit")}
    test_units = {int(uid): g.reset_index(drop=True) for uid, g in test_df.groupby("unit")}

    return CMAPSSDataset(
        train_units=train_units,
        test_units=test_units,
        feature_columns=feature_columns,
    )


def build_windows(
    units: dict[int, pd.DataFrame],
    feature_columns: list[str],
    means: np.ndarray | None = None,
    stds: np.ndarray | None = None,
    return_unit_ids: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows from per-unit cycle data.

    Args:
        units: as returned by `load_cmapss_fd001` for train or test split.
        feature_columns: sensor columns to include.
        means / stds: optional per-sensor normalization stats. If None,
            computed from the input units (used for the train split).

    Returns:
        windows: (N, WINDOW_LENGTH, F) float32 array.
        targets: (N,) float32 RUL targets for the last cycle of each window.
        unit_ids: (N,) int unit-id for each window.
        end_cycles: (N,) int cycle number at the end of each window.
    """

    windows: list[np.ndarray] = []
    targets: list[float] = []
    unit_ids: list[int] = []
    end_cycles: list[int] = []

    if means is None or stds is None:
        all_features = np.concatenate(
            [u[feature_columns].to_numpy(dtype=np.float32) for u in units.values()],
            axis=0,
        )
        means = all_features.mean(axis=0)
        stds = all_features.std(axis=0) + 1e-6

    for unit_id, unit_df in units.items():
        feat = unit_df[feature_columns].to_numpy(dtype=np.float32)
        rul = unit_df["rul"].to_numpy(dtype=np.float32)
        n = feat.shape[0]
        feat = (feat - means) / stds
        if n < WINDOW_LENGTH:
            # Pad short test sequences from the start with the first row repeated.
            pad_count = WINDOW_LENGTH - n
            feat = np.concatenate([np.tile(feat[:1], (pad_count, 1)), feat], axis=0)
            rul = np.concatenate([np.full(pad_count, rul[0], dtype=np.float32), rul])
            n = feat.shape[0]
        for start in range(0, n - WINDOW_LENGTH + 1):
            end = start + WINDOW_LENGTH
            windows.append(feat[start:end])
            targets.append(float(rul[end - 1]))
            unit_ids.append(unit_id)
            end_cycles.append(int(unit_df["cycle"].iloc[min(end - 1, len(unit_df) - 1)]))

    return (
        np.stack(windows, axis=0),
        np.asarray(targets, dtype=np.float32),
        np.asarray(unit_ids, dtype=np.int32),
        np.asarray(end_cycles, dtype=np.int32),
    )


def last_window_per_test_unit(
    units: dict[int, pd.DataFrame],
    feature_columns: list[str],
    means: np.ndarray,
    stds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one window per test unit, ending at the last available cycle.

    These are the windows whose predicted RUL should match RUL_FD001.txt.
    """

    windows: list[np.ndarray] = []
    targets: list[float] = []
    unit_ids: list[int] = []

    for unit_id, unit_df in units.items():
        feat = unit_df[feature_columns].to_numpy(dtype=np.float32)
        rul = unit_df["rul"].to_numpy(dtype=np.float32)
        feat = (feat - means) / stds
        n = feat.shape[0]
        if n < WINDOW_LENGTH:
            pad = WINDOW_LENGTH - n
            feat = np.concatenate([np.tile(feat[:1], (pad, 1)), feat], axis=0)
            rul = np.concatenate([np.full(pad, rul[0], dtype=np.float32), rul])
        windows.append(feat[-WINDOW_LENGTH:])
        targets.append(float(rul[-1]))
        unit_ids.append(unit_id)

    return (
        np.stack(windows, axis=0),
        np.asarray(targets, dtype=np.float32),
        np.asarray(unit_ids, dtype=np.int32),
    )


def cmapss_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Asymmetric CMAPSS scoring metric: late predictions penalized harder.

    From Saxena et al. (2008). Lower is better. Score = sum over samples of
    exp(d/13) - 1 if d > 0 else exp(-d/10) - 1, where d = pred - target.
    """

    d = predictions - targets
    return float(np.where(d >= 0, np.exp(d / 13.0) - 1, np.exp(-d / 10.0) - 1).sum())
