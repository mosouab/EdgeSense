"""Evaluation utilities for anomaly detection on failure intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

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


@dataclass(frozen=True)
class PointAdjustedMetrics:
    """Point-adjusted precision/recall/F1 metrics."""

    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class PRThresholdResult:
    """Evaluation artifacts using PR-optimized thresholding."""

    scores: np.ndarray
    labels: np.ndarray
    predictions: np.ndarray
    adjusted_predictions: np.ndarray
    threshold: float
    metrics: EvaluationMetrics
    adjusted_metrics: PointAdjustedMetrics


def apply_median_filter(scores: np.ndarray, window_size: int) -> np.ndarray:
    """Apply a median filter to 1D anomaly scores.

    Args:
        scores: Array of anomaly scores.
        window_size: Odd-sized median window.

    Returns:
        Smoothed scores array with the same length.
    """

    if scores.ndim != 1:
        raise ValueError("scores must be a 1D array.")
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")

    pad = window_size // 2
    padded = np.pad(scores, (pad, pad), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_size)
    return np.median(windows, axis=1)


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


def compute_optimal_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute the threshold that maximizes F1 from the PR curve."""

    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same shape.")

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        raise ValueError("Cannot compute thresholds for PR curve.")

    f1_scores = _f1_from_pr(precision, recall)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


def evaluate_anomaly_detection_pr(
    model: USADConv1d,
    windows: np.ndarray,
    start_times: pd.Series,
    end_times: pd.Series,
    failure_reports: pd.DataFrame,
    scoring_config: ScoringConfig,
) -> PRThresholdResult:
    """Evaluate anomaly detection using PR-optimized thresholding.

    Args:
        model: Trained USADConv1d model.
        windows: Windowed dataset of shape (num_windows, window_size, num_features).
        start_times: Window start timestamps.
        end_times: Window end timestamps.
        failure_reports: Failure intervals to label anomalies.
        scoring_config: Scoring configuration.

    Returns:
        PRThresholdResult containing raw and point-adjusted metrics.
    """

    labels = label_windows_by_failures(start_times, end_times, failure_reports)
    scores = compute_usad_scores(model, windows, scoring_config)

    threshold = compute_optimal_f1_threshold(labels, scores)
    predictions = scores >= threshold
    adjusted_predictions = point_adjust_predictions(labels, predictions)

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

    adjusted_metrics = PointAdjustedMetrics(
        precision=float(precision_score(labels, adjusted_predictions, zero_division=0)),
        recall=float(recall_score(labels, adjusted_predictions, zero_division=0)),
        f1=float(f1_score(labels, adjusted_predictions, zero_division=0)),
    )

    return PRThresholdResult(
        scores=scores,
        labels=labels,
        predictions=predictions,
        adjusted_predictions=adjusted_predictions,
        threshold=threshold,
        metrics=metrics,
        adjusted_metrics=adjusted_metrics,
    )


def apply_temporal_persistence(
    predictions: np.ndarray,
    min_consecutive: int,
) -> np.ndarray:
    """Apply a time-to-trigger filter requiring consecutive anomalies.

    A window is flagged as anomalous only if it belongs to a run of
    at least `min_consecutive` consecutive positive predictions.
    """

    if min_consecutive <= 0:
        raise ValueError("min_consecutive must be a positive integer.")
    if predictions.ndim != 1:
        raise ValueError("predictions must be a 1D array.")

    filtered = np.zeros_like(predictions, dtype=bool)
    run_start = None
    run_length = 0

    for idx, is_anomaly in enumerate(predictions):
        if is_anomaly:
            if run_start is None:
                run_start = idx
                run_length = 1
            else:
                run_length += 1

            if run_length >= min_consecutive and run_start is not None:
                filtered[run_start : idx + 1] = True
        else:
            run_start = None
            run_length = 0

    return filtered


def point_adjust_predictions(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Apply USAD-style point adjustment over contiguous anomaly segments."""

    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape.")

    adjusted = predictions.copy()
    idx = 0
    while idx < len(labels):
        if labels[idx]:
            start = idx
            while idx < len(labels) and labels[idx]:
                idx += 1
            end = idx
            if predictions[start:end].any():
                adjusted[start:end] = True
        else:
            idx += 1
    return adjusted


def _f1_from_pr(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    """Compute F1 scores aligned with PR curve thresholds."""

    precision = precision[:-1]
    recall = recall[:-1]
    return (2 * precision * recall) / (precision + recall + 1e-12)
