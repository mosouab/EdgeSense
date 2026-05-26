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
]
