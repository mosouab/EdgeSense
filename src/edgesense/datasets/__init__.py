"""Dataset adapters for predictive-maintenance benchmarks.

Each module here is a thin loader for one public dataset. They expose the
asset and label data in a form that the rest of edgesense (model,
training, scoring, evaluation) can consume without knowing which dataset
the inputs came from.

Available adapters:
- metropt: Metro do Porto air-compressor dataset (continuous time series,
  air-leak failure intervals). Detection-style evaluation.
- hydraulic: UCI Condition Monitoring of Hydraulic Systems (per-cycle
  instances, multi-component fault labels). Multi-fault detection.
- cmapss: NASA CMAPSS turbofan degradation dataset (run-to-failure cycles
  with RUL ground truth). RUL regression.
"""

from .metropt import (
    MetroPTDataset,
    load_metropt_dataset,
    load_metropt_failures,
)
from .hydraulic import (
    HydraulicDataset,
    NOMINAL_VALUES,
    component_split,
    load_hydraulic_dataset,
)

__all__ = [
    "MetroPTDataset",
    "load_metropt_dataset",
    "load_metropt_failures",
    "HydraulicDataset",
    "NOMINAL_VALUES",
    "component_split",
    "load_hydraulic_dataset",
]
