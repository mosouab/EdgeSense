"""Data ingestion utilities for the Metro.PT Air Compressor dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_METROPT_CSV = Path(__file__).resolve().parents[2] / "data" / "MetroPT3(AirCompressor).csv"
DEFAULT_TIMESTAMP_COL = "timestamp"
EXPECTED_FEATURE_COLUMNS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]


@dataclass(frozen=True)
class MetroPTDataset:
    """Container holding the loaded Metro.PT dataset and metadata.

    Attributes:
        data: Full dataset with timestamp and feature columns.
        feature_columns: Ordered list of feature column names.
        timestamp_col: Name of the timestamp column.
        sampling_interval_seconds: Median sampling interval in seconds.
        start_time: Timestamp of the first record.
        end_time: Timestamp of the last record.
    """

    data: pd.DataFrame
    feature_columns: list[str]
    timestamp_col: str
    sampling_interval_seconds: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp


def load_metropt_dataset(
    csv_path: Path = DEFAULT_METROPT_CSV,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
) -> MetroPTDataset:
    """Load and validate the Metro.PT Air Compressor dataset.

    Args:
        csv_path: Path to the Metro.PT CSV file.
        timestamp_col: Name of the timestamp column.

    Returns:
        MetroPTDataset containing the cleaned dataframe and metadata.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing or timestamps are invalid.
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"Metro.PT CSV not found at {csv_path}")

    data = pd.read_csv(csv_path)
    data = _drop_unnamed_columns(data)

    if timestamp_col not in data.columns:
        raise ValueError(f"Missing required timestamp column '{timestamp_col}'.")

    missing = sorted(set(EXPECTED_FEATURE_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="raise")
    data = data.sort_values(timestamp_col).reset_index(drop=True)

    if not data[timestamp_col].is_monotonic_increasing:
        raise ValueError("Timestamp column is not monotonic increasing after sorting.")

    feature_columns = [col for col in EXPECTED_FEATURE_COLUMNS if col in data.columns]
    data[feature_columns] = data[feature_columns].apply(pd.to_numeric, errors="raise")

    sampling_interval_seconds = _infer_sampling_interval_seconds(data[timestamp_col])

    return MetroPTDataset(
        data=data,
        feature_columns=feature_columns,
        timestamp_col=timestamp_col,
        sampling_interval_seconds=sampling_interval_seconds,
        start_time=data[timestamp_col].iloc[0],
        end_time=data[timestamp_col].iloc[-1],
    )


def load_failure_reports() -> pd.DataFrame:
    """Load failure intervals described in the Metro.PT data documentation.

    Returns:
        DataFrame with failure intervals and metadata for evaluation.
    """

    reports = [
        {
            "failure_id": 1,
            "start_time": "2020-04-18 00:00",
            "end_time": "2020-04-18 23:59",
            "failure_type": "AirLeak",
            "severity": "High stress",
            "report": "Air leak",
        },
        {
            "failure_id": 2,
            "start_time": "2020-05-29 23:30",
            "end_time": "2020-05-30 06:00",
            "failure_type": "AirLeak",
            "severity": "High stress",
            "report": "Maintenance on 30 Apr at 12:00",
        },
        {
            "failure_id": 3,
            "start_time": "2020-06-05 10:00",
            "end_time": "2020-06-07 14:30",
            "failure_type": "AirLeak",
            "severity": "High stress",
            "report": "Maintenance on 8 Jun at 16:00",
        },
        {
            "failure_id": 4,
            "start_time": "2020-07-15 14:30",
            "end_time": "2020-07-15 19:00",
            "failure_type": "AirLeak",
            "severity": "High stress",
            "report": "Maintenance on 16 Jul at 00:00",
        },
    ]

    failures = pd.DataFrame(reports)
    failures["start_time"] = pd.to_datetime(failures["start_time"], errors="raise")
    failures["end_time"] = pd.to_datetime(failures["end_time"], errors="raise")
    return failures


def _drop_unnamed_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Drop unnamed index-like columns created by CSV exports."""

    unnamed = [col for col in data.columns if not str(col).strip() or str(col).startswith("Unnamed")]
    if unnamed:
        return data.drop(columns=unnamed)
    return data


def _infer_sampling_interval_seconds(timestamps: pd.Series) -> float:
    """Infer the median sampling interval in seconds from a timestamp series."""

    deltas = timestamps.diff().dropna().dt.total_seconds()
    if deltas.empty:
        return float("nan")
    return float(deltas.median())
