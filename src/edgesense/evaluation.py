"""Evaluation utilities for anomaly detection on failure intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .models import USADConv1d
from .scoring import ScoringConfig, ThresholdConfig, compute_threshold, compute_usad_scores, flag_anomalies


@dataclass(frozen=True)
class EvaluationMetrics:
    """Window-level evaluation metrics."""

    precision: float
    recall: float
    f1: float
    accuracy: float
    auc: float | None
    confusion_matrix: np.ndarray


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation artifacts for anomaly detection."""

    scores: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    threshold: float
    metrics: EvaluationMetrics


def label_windows_by_failures(
    start_times: pd.Series,
    end_times: pd.Series,
    failure_reports: pd.DataFrame,
) -> np.ndarray:
    """Label windows as anomalous if they overlap any failure interval.

    Args:
        start_times: Start timestamps for each window.
        end_times: End timestamps for each window.
        failure_reports: Failure intervals with start_time and end_time columns.

    Returns:
        Boolean array where True indicates an anomalous window.
    """

    if start_times.shape[0] != end_times.shape[0]:
        raise ValueError("start_times and end_times must have the same length.")
    if failure_reports.empty:
        return np.zeros(start_times.shape[0], dtype=bool)

    start_series = pd.to_datetime(start_times, errors="raise")
    end_series = pd.to_datetime(end_times, errors="raise")
    labels = np.zeros(start_series.shape[0], dtype=bool)

    for _, row in failure_reports.iterrows():
        failure_start = pd.to_datetime(row["start_time"], errors="raise")
        failure_end = pd.to_datetime(row["end_time"], errors="raise")
        overlaps = (start_series <= failure_end) & (end_series >= failure_start)
        labels = labels | overlaps.to_numpy()

    return labels


def evaluate_anomaly_detection(
    model: USADConv1d,
    windows: np.ndarray,
    start_times: pd.Series,
    end_times: pd.Series,
    failure_reports: pd.DataFrame,
    scoring_config: ScoringConfig,
    threshold_config: ThresholdConfig,
) -> EvaluationResult:
    """Evaluate anomaly detection against failure intervals.

    Args:
        model: Trained USADConv1d model.
        windows: Windowed dataset of shape (num_windows, window_size, num_features).
        start_times: Window start timestamps.
        end_times: Window end timestamps.
        failure_reports: Failure intervals to label anomalies.
        scoring_config: Scoring configuration.
        threshold_config: Threshold configuration.

    Returns:
        EvaluationResult with scores, labels, threshold, and metrics.
    """

    labels = label_windows_by_failures(start_times, end_times, failure_reports)
    scores = compute_usad_scores(model, windows, scoring_config)

    healthy_scores = scores[~labels]
    if healthy_scores.size == 0:
        raise ValueError("No healthy windows available to compute threshold.")

    threshold = compute_threshold(healthy_scores, threshold_config)
    predictions = flag_anomalies(scores, threshold)

    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    confusion = confusion_matrix(labels, predictions)
    auc = None
    if labels.any() and (~labels).any():
        auc = roc_auc_score(labels, scores)

    metrics = EvaluationMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        accuracy=float(accuracy),
        auc=float(auc) if auc is not None else None,
        confusion_matrix=confusion,
    )

    return EvaluationResult(
        scores=scores,
        labels=labels,
        predictions=predictions,
        threshold=threshold,
        metrics=metrics,
    )
