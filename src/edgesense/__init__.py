"""Core package for the EdgeSense project."""

from .data_ingestion import MetroPTDataset, load_failure_reports, load_metropt_dataset
from .preprocessing import MetroPTPreprocessor, PreprocessingArtifacts, build_healthy_mask
from .windowing import (
    WindowConfig,
    WindowedDataset,
    build_window_mask,
    build_window_start_indices,
    compute_num_windows,
    create_sliding_windows,
    iter_sliding_windows,
)
from .models import USADConv1d, USADConv1dConfig
from .training import (
    EarlyStoppingConfig,
    TrainingConfig,
    TrainingHistory,
    create_dataloader,
    seed_all,
    split_train_validation,
    train_usad,
    train_usad_with_validation,
)
from .scoring import ScoringConfig, ThresholdConfig, compute_threshold, compute_usad_scores, flag_anomalies
from .evaluation import EvaluationMetrics, EvaluationResult, evaluate_anomaly_detection, label_windows_by_failures

__all__ = [
    "MetroPTDataset",
    "load_failure_reports",
    "load_metropt_dataset",
    "MetroPTPreprocessor",
    "PreprocessingArtifacts",
    "build_healthy_mask",
    "WindowConfig",
    "WindowedDataset",
    "build_window_mask",
    "build_window_start_indices",
    "compute_num_windows",
    "create_sliding_windows",
    "iter_sliding_windows",
    "USADConv1d",
    "USADConv1dConfig",
    "TrainingConfig",
    "TrainingHistory",
    "EarlyStoppingConfig",
    "create_dataloader",
    "seed_all",
    "split_train_validation",
    "train_usad",
    "train_usad_with_validation",
    "ScoringConfig",
    "ThresholdConfig",
    "compute_threshold",
    "compute_usad_scores",
    "flag_anomalies",
    "EvaluationMetrics",
    "EvaluationResult",
    "evaluate_anomaly_detection",
    "label_windows_by_failures",
]
