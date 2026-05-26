from __future__ import annotations

from pathlib import Path
import sys

SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from edgesense.data_ingestion import load_failure_reports, load_metropt_dataset
from edgesense.preprocessing import MetroPTPreprocessor, build_healthy_mask


def main() -> None:
    dataset = load_metropt_dataset()
    failures = load_failure_reports()
    preprocessor = MetroPTPreprocessor(
        feature_columns=dataset.feature_columns,
        timestamp_col=dataset.timestamp_col,
    )
    scaled_features = preprocessor.fit_transform(dataset, failures)
    healthy_mask = build_healthy_mask(dataset.data, dataset.timestamp_col, failures)

    print(
        "Loaded Metro.PT dataset:"
        f" rows={len(dataset.data)},"
        f" features={len(dataset.feature_columns)},"
        f" start={dataset.start_time},"
        f" end={dataset.end_time},"
        f" median_dt={dataset.sampling_interval_seconds:.2f}s"
    )
    print(
        "Preprocessing:"
        f" scaled_shape={scaled_features.shape},"
        f" healthy_rows={healthy_mask.sum()},"
        f" unhealthy_rows={(~healthy_mask).sum()}"
    )
    print(f"Failure reports: {len(failures)} intervals")


if __name__ == "__main__":
    main()
