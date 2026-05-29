"""Health Score: a 0-100 indicator derived from the anomaly score.

The score is intuitive for an operator: 100 means "behaves like the
training-period healthy baseline", 0 means "at or above the deploy
threshold". Inside the band, it interpolates linearly.
"""

from __future__ import annotations

import numpy as np


def health_score(
    scores: np.ndarray,
    healthy_reference: np.ndarray,
    alert_threshold: float,
    healthy_quantile: float = 95.0,
) -> np.ndarray:
    """Convert anomaly scores to a 0-100 Health Score.

    Args:
        scores: array of smoothed anomaly scores to convert.
        healthy_reference: scores collected during a known-healthy period
            (training set or on-site recalibration window). Used to set
            the "fully healthy" anchor of the band.
        alert_threshold: the score level at which the system fires an
            alert. Maps to health = 0.
        healthy_quantile: the percentile of `healthy_reference` mapped to
            health = 100. Defaults to 95 so the bottom 5% jittery samples
            don't pull the band.

    Returns:
        Array of health scores in [0, 100].
    """

    if scores.ndim != 1:
        raise ValueError("scores must be 1D.")
    if healthy_reference.size == 0:
        raise ValueError("healthy_reference is empty.")
    if not 0 < healthy_quantile < 100:
        raise ValueError("healthy_quantile must be in (0, 100).")

    healthy_anchor = float(np.percentile(healthy_reference, healthy_quantile))
    if alert_threshold <= healthy_anchor:
        # Degenerate ordering; fall back to binary 100/0.
        return np.where(scores >= alert_threshold, 0.0, 100.0).astype(np.float32)

    raw = 100.0 * (alert_threshold - scores) / (alert_threshold - healthy_anchor)
    return np.clip(raw, 0.0, 100.0).astype(np.float32)
