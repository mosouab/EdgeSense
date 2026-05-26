"""Preprocessing utilities for Metro.PT time-series data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data_ingestion import MetroPTDataset


@dataclass(frozen=True)
class PreprocessingArtifacts:
    """Artifacts produced by preprocessing and scaling.

    Attributes:
        scaler: Fitted feature scaler.
        feature_columns: Ordered list of feature column names.
        timestamp_col: Name of the timestamp column.
    """

    scaler: StandardScaler
    feature_columns: list[str]
    timestamp_col: str


class MetroPTPreprocessor:
    """Preprocess and scale Metro.PT features for anomaly detection."""

    def __init__(self, feature_columns: list[str], timestamp_col: str) -> None:
        self._feature_columns = feature_columns
        self._timestamp_col = timestamp_col
        self._scaler = StandardScaler()

    @property
    def artifacts(self) -> PreprocessingArtifacts:
        """Return preprocessing artifacts for downstream steps."""

        return PreprocessingArtifacts(
            scaler=self._scaler,
            feature_columns=self._feature_columns,
            timestamp_col=self._timestamp_col,
        )

    def fit(self, dataset: MetroPTDataset, failure_reports: pd.DataFrame | None = None) -> None:
        """Fit the scaler using healthy baseline data only.

        Args:
            dataset: Loaded Metro.PT dataset.
            failure_reports: Optional failure intervals to exclude from fitting.
        """

        features = _impute_missing_values(dataset.data, self._feature_columns)
        healthy_mask = build_healthy_mask(dataset.data, dataset.timestamp_col, failure_reports)
        healthy_features = features.loc[healthy_mask]
        if healthy_features.empty:
            raise ValueError("No healthy samples available to fit the scaler.")
        self._scaler.fit(healthy_features.values)

    def transform(self, dataset: MetroPTDataset) -> pd.DataFrame:
        """Scale all feature rows using the fitted scaler.

        Args:
            dataset: Loaded Metro.PT dataset.

        Returns:
            DataFrame of scaled features with original indexing.
        """

        features = _impute_missing_values(dataset.data, self._feature_columns)
        scaled = self._scaler.transform(features.values)
        return pd.DataFrame(scaled, columns=self._feature_columns, index=features.index)

    def fit_transform(
        self,
        dataset: MetroPTDataset,
        failure_reports: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Fit the scaler on healthy data and transform the full dataset."""

        self.fit(dataset, failure_reports)
        return self.transform(dataset)

    def save(self, path: Path) -> None:
        """Persist preprocessing artifacts to disk.

        Args:
            path: Target file path for serialization.
        """

        payload = {
            "scaler": self._scaler,
            "feature_columns": self._feature_columns,
            "timestamp_col": self._timestamp_col,
        }
        with path.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: Path) -> "MetroPTPreprocessor":
        """Load preprocessing artifacts from disk.

        Args:
            path: Serialized artifacts file path.

        Returns:
            MetroPTPreprocessor with restored scaler and metadata.
        """

        with path.open("rb") as handle:
            payload = pickle.load(handle)
        instance = cls(
            feature_columns=payload["feature_columns"],
            timestamp_col=payload["timestamp_col"],
        )
        instance._scaler = payload["scaler"]
        return instance


def build_healthy_mask(
    data: pd.DataFrame,
    timestamp_col: str,
    failure_reports: pd.DataFrame | None,
) -> pd.Series:
    """Build a boolean mask for healthy rows based on failure intervals.

    Args:
        data: Dataset containing a timestamp column.
        timestamp_col: Name of the timestamp column.
        failure_reports: DataFrame with start_time/end_time columns.

    Returns:
        Boolean Series where True indicates healthy rows.
    """

    timestamps = pd.to_datetime(data[timestamp_col], errors="raise")
    healthy_mask = pd.Series(True, index=data.index)
    if failure_reports is None or failure_reports.empty:
        return healthy_mask

    for _, row in failure_reports.iterrows():
        start_time = pd.to_datetime(row["start_time"], errors="raise")
        end_time = pd.to_datetime(row["end_time"], errors="raise")
        in_failure = timestamps.between(start_time, end_time, inclusive="both")
        healthy_mask = healthy_mask & ~in_failure

    return healthy_mask


def _impute_missing_values(data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Fill missing sensor values using linear interpolation.

    The interpolation maintains temporal continuity so that the scaler does not
    receive NaNs, while minimizing distortion to the original signal.
    """

    features = data[feature_columns].copy()
    features = features.interpolate(method="linear", limit_direction="both")
    features = features.ffill().bfill()
    if features.isna().any().any():
        raise ValueError("Missing values remain after interpolation.")
    return features
