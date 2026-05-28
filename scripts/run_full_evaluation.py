"""Train USAD, score the eval split, and save model + metric artifacts.

Plot generation is delegated to scripts/generate_figures.py, which this script
invokes at the end.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))
# Allow `from generate_figures import ...` below.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from edgesense.data_ingestion import MetroPTDataset, load_failure_reports, load_metropt_dataset
from edgesense.evaluation import (
    apply_median_filter,
    apply_temporal_persistence,
    compute_optimal_f1_threshold,
    point_adjust_predictions,
    label_windows_by_failures,
)
from edgesense.models import USADConv1d, USADConv1dConfig
from edgesense.preprocessing import MetroPTPreprocessor
from edgesense.scoring import ScoringConfig, compute_usad_scores
from edgesense.training import EarlyStoppingConfig, TrainingConfig, seed_all, train_usad
from edgesense.windowing import create_sliding_windows

# Temporal split: train on pre-failure data, evaluate on the rest.
# All four labeled failures occur on or after 2020-04-18, so both the training
# range and the recalibration window are guaranteed-healthy.
TRAIN_END = pd.Timestamp("2020-04-01")
# Simulate on-site recalibration: the first RECAL_DAYS of the eval period
# are used to refit the threshold against the deployment-time score distribution
# (which can drift from the training-period distribution).
RECAL_END = pd.Timestamp("2020-04-15")
# Quantile of recal-window scores used as the deployable threshold.
# We use p99 rather than p99.9 because the deployment-time score distribution
# has heavy tails that a more extreme quantile would chase into a single
# outlier window, producing a degenerately strict threshold.
HEALTHY_QUANTILE = 99.0


def main() -> None:
    """Train USAD, evaluate, and generate artifacts."""

    print("--- EdgeSense Full Evaluation Pipeline ---")

    output_dir = Path("reports") / "full_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading data and applying temporal train/eval split...")
    dataset = load_metropt_dataset()
    failures = load_failure_reports()

    train_dataset, eval_dataset = split_dataset_temporally(dataset, TRAIN_END)
    print(
        f"  train: {train_dataset.start_time} -> {train_dataset.end_time} "
        f"({len(train_dataset.data):,} rows)"
    )
    print(
        f"  eval:  {eval_dataset.start_time} -> {eval_dataset.end_time} "
        f"({len(eval_dataset.data):,} rows)"
    )

    print("[2/6] Fitting preprocessor on training-period healthy data...")
    preprocessor = MetroPTPreprocessor(
        feature_columns=dataset.feature_columns,
        timestamp_col=dataset.timestamp_col,
    )
    preprocessor.fit(train_dataset, failures)
    scaled_train = preprocessor.transform(train_dataset)
    scaled_eval = preprocessor.transform(eval_dataset)
    preprocessor.save(output_dir / "preprocessor.pkl")

    window_size = 100
    stride = 50

    print("[3/6] Building sliding windows...")
    train_windows = create_sliding_windows(
        scaled_train,
        window_size=window_size,
        stride=stride,
        timestamps=train_dataset.data[dataset.timestamp_col],
    )
    eval_windows = create_sliding_windows(
        scaled_eval,
        window_size=window_size,
        stride=stride,
        timestamps=eval_dataset.data[dataset.timestamp_col],
    )

    model_config = USADConv1dConfig(
        in_features=scaled_train.shape[1],
        base_channels=32,
        latent_channels=64,
        downsample_layers=2,
    )
    # Seed before model construction so weight initialization is reproducible
    # regardless of what RNG-consuming operations ran upstream in this script.
    seed_all(42)
    model = USADConv1d(model_config)

    print("[4/6] Training USAD with adversarial schedule + early stopping...")
    train_config = TrainingConfig(
        batch_size=256,
        epochs=50,
        learning_rate=1e-3,
        adv_ramp_epochs=30,
        adv_max_weight=0.3,
        grad_clip_norm=1.0,
    )
    stop_config = EarlyStoppingConfig(patience=12, min_delta=1e-4, max_epochs=60, val_fraction=0.1)
    from edgesense.training import split_train_validation
    train_only, val_only = split_train_validation(train_windows.windows, stop_config.val_fraction)
    history = train_usad(
        model,
        train_only,
        train_config,
        val_windows=val_only,
        early_stopping=stop_config,
    )

    torch.save(model.state_dict(), output_dir / "usad_conv1d.pt")
    with (output_dir / "model_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(model_config), handle, indent=2)

    print("[5/6] Scoring train and eval windows...")
    scoring_config = ScoringConfig(alpha=0.3, beta=0.7, batch_size=512)
    train_scores_raw = compute_usad_scores(model, train_windows.windows, scoring_config)
    eval_scores_raw = compute_usad_scores(model, eval_windows.windows, scoring_config)

    train_scores_smoothed = apply_median_filter(train_scores_raw, window_size=11)
    eval_scores_smoothed = apply_median_filter(eval_scores_raw, window_size=11)

    labels = label_windows_by_failures(
        eval_windows.start_times,
        eval_windows.end_times,
        failures,
    )

    # Split eval windows into a deployment-time recalibration window
    # (first RECAL_END days) and the held-out test horizon. The recal window
    # ends before the first labeled failure (2020-04-18), so it is healthy
    # by report and never sees a fault during threshold-setting.
    eval_window_starts = pd.to_datetime(eval_windows.start_times, errors="raise")
    eval_window_ends = pd.to_datetime(eval_windows.end_times, errors="raise")
    recal_mask = eval_window_ends.lt(RECAL_END).to_numpy()
    test_mask = eval_window_starts.ge(RECAL_END).to_numpy()

    if not recal_mask.any() or not test_mask.any():
        raise ValueError("Recalibration split produced an empty partition.")

    recal_scores = eval_scores_smoothed[recal_mask]
    test_scores = eval_scores_smoothed[test_mask]
    test_labels = labels[test_mask]

    # Headline (deployable) threshold: high quantile of healthy scores
    # collected from the first RECAL_END - eval_start days on-site.
    # No failure labels are consulted to set it.
    recalibrated_threshold = float(np.percentile(recal_scores, HEALTHY_QUANTILE))

    # Reference 1: original training-period threshold (no on-site recalibration).
    # Useful to quantify how much the deployment-time refit is buying us.
    training_threshold = float(np.percentile(train_scores_smoothed, HEALTHY_QUANTILE))

    # Reference 2 (oracle): PR-optimal threshold on the test labels.
    # Upper bound on what label-free thresholding can achieve.
    oracle_threshold = compute_optimal_f1_threshold(test_labels, test_scores)

    print(
        f"  recalibrated threshold (p{HEALTHY_QUANTILE} of first {(RECAL_END - eval_dataset.start_time).days} days on-site): {recalibrated_threshold:.6f}"
    )
    print(
        f"  training-period threshold (no recalibration):                 {training_threshold:.6f}"
    )
    print(
        f"  oracle PR-optimal threshold (reference):                      {oracle_threshold:.6f}"
    )

    print("[6/6] Computing metrics and saving artifacts...")
    metrics_payload = {
        "thresholds": {
            "headline_recalibrated": recalibrated_threshold,
            "supplementary_training_quantile": training_threshold,
            "reference_oracle_pr_optimal": oracle_threshold,
            "headline_quantile": HEALTHY_QUANTILE,
        },
        "headline_recalibrated": _build_metric_block(
            test_labels, test_scores, recalibrated_threshold
        ),
        "supplementary_training_quantile": _build_metric_block(
            test_labels, test_scores, training_threshold
        ),
        "reference_oracle_pr_optimal": _build_metric_block(
            test_labels, test_scores, oracle_threshold
        ),
        "training": {
            "epochs": len(history.ae1_losses),
            "final_ae1_loss": history.ae1_losses[-1],
            "final_ae2_loss": history.ae2_losses[-1],
            "final_val_recon_loss": history.val_recon_losses[-1] if history.val_recon_losses else None,
        },
        "split": {
            "train_end": TRAIN_END.isoformat(),
            "train_start": train_dataset.start_time.isoformat(),
            "recal_start": eval_dataset.start_time.isoformat(),
            "recal_end": RECAL_END.isoformat(),
            "test_start": RECAL_END.isoformat(),
            "test_end": eval_dataset.end_time.isoformat(),
            "train_rows": int(len(train_dataset.data)),
            "eval_rows": int(len(eval_dataset.data)),
        },
        "windowing": {"window_size": window_size, "stride": stride},
        "score_smoothing": {"median_window": 11},
        "temporal_persistence": {"min_consecutive": 25},
        "train_windows": int(train_windows.windows.shape[0]),
        "recal_windows": int(recal_mask.sum()),
        "test_windows": int(test_mask.sum()),
        "eval_windows": int(eval_windows.windows.shape[0]),
        "adversarial_schedule": {
            "adv_ramp_epochs": train_config.adv_ramp_epochs,
            "adv_max_weight": train_config.adv_max_weight,
            "grad_clip_norm": train_config.grad_clip_norm,
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    # Persist the full per-window timeline so figures can highlight the recal
    # window. Predictions are computed using the recalibrated threshold; rows
    # inside the recal window are flagged via the `in_recalibration` column.
    in_recal = recal_mask
    predictions = eval_scores_smoothed >= recalibrated_threshold
    adjusted_predictions = point_adjust_predictions(labels, predictions)
    persistence_predictions = apply_temporal_persistence(predictions, min_consecutive=25)
    persistence_adjusted = point_adjust_predictions(labels, persistence_predictions)

    save_training_history(history, output_dir / "training_history.csv")
    save_score_timeline(
        eval_windows.start_times,
        eval_windows.end_times,
        eval_scores_raw,
        eval_scores_smoothed,
        labels,
        predictions,
        adjusted_predictions,
        persistence_predictions,
        persistence_adjusted,
        in_recal,
        output_dir / "scores_timeline.csv",
    )
    save_pr_curve_csv(test_labels, test_scores, output_dir / "pr_curve.csv")

    print(f"Saved evaluation artifacts to: {output_dir.resolve()}")
    print("Generating figures...")
    from generate_figures import main as generate_figures_main
    generate_figures_main()


def split_dataset_temporally(
    dataset: MetroPTDataset, train_end: pd.Timestamp
) -> tuple[MetroPTDataset, MetroPTDataset]:
    """Split a MetroPTDataset into two by timestamp.

    The split point belongs to the eval partition (train uses strictly `<`).
    """

    timestamps = pd.to_datetime(dataset.data[dataset.timestamp_col], errors="raise")
    train_mask = timestamps < train_end
    train_df = dataset.data.loc[train_mask].reset_index(drop=True)
    eval_df = dataset.data.loc[~train_mask].reset_index(drop=True)

    if train_df.empty or eval_df.empty:
        raise ValueError(f"Temporal split at {train_end} yields an empty partition.")

    return (
        MetroPTDataset(
            data=train_df,
            feature_columns=dataset.feature_columns,
            timestamp_col=dataset.timestamp_col,
            sampling_interval_seconds=dataset.sampling_interval_seconds,
            start_time=train_df[dataset.timestamp_col].iloc[0],
            end_time=train_df[dataset.timestamp_col].iloc[-1],
        ),
        MetroPTDataset(
            data=eval_df,
            feature_columns=dataset.feature_columns,
            timestamp_col=dataset.timestamp_col,
            sampling_interval_seconds=dataset.sampling_interval_seconds,
            start_time=eval_df[dataset.timestamp_col].iloc[0],
            end_time=eval_df[dataset.timestamp_col].iloc[-1],
        ),
    )


def _build_metric_block(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict:
    """Build the {raw, point_adjusted, persistence, persistence_point_adjusted} block."""

    predictions = scores >= threshold
    adjusted = point_adjust_predictions(labels, predictions)
    persistence = apply_temporal_persistence(predictions, min_consecutive=25)
    persistence_adjusted = point_adjust_predictions(labels, persistence)

    return {
        "raw": compute_metrics(labels, predictions, scores),
        "point_adjusted": compute_metrics(labels, adjusted, scores, include_auc=False),
        "persistence": compute_metrics(labels, persistence, scores, include_auc=False),
        "persistence_point_adjusted": compute_metrics(
            labels, persistence_adjusted, scores, include_auc=False
        ),
    }


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    include_auc: bool = True,
) -> dict[str, float | None]:
    """Compute precision/recall/F1/accuracy and optional AUC."""

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    auc = None
    if include_auc and labels.any() and (~labels).any():
        auc = roc_auc_score(labels, scores)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "auc": float(auc) if auc is not None else None,
    }


def save_training_history(history, output_path: Path) -> None:
    """Save training history to CSV."""

    data = {
        "epoch": list(range(1, len(history.ae1_losses) + 1)),
        "ae1_loss": history.ae1_losses,
        "ae2_loss": history.ae2_losses,
        "val_recon_loss": history.val_recon_losses or [],
    }
    pd.DataFrame(data).to_csv(output_path, index=False)


def save_pr_curve_csv(
    labels: np.ndarray,
    scores: np.ndarray,
    csv_path: Path,
) -> None:
    """Save PR curve data to CSV (plotting is handled by generate_figures.py)."""

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.append(thresholds, np.nan),
        }
    ).to_csv(csv_path, index=False)


def save_score_timeline(
    start_times: pd.Series,
    end_times: pd.Series,
    scores_raw: np.ndarray,
    scores_smoothed: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    adjusted_predictions: np.ndarray,
    persistence_predictions: np.ndarray,
    persistence_adjusted: np.ndarray,
    in_recalibration: np.ndarray,
    output_path: Path,
) -> None:
    """Save scores and labels per window to CSV."""

    start_series = pd.to_datetime(start_times, errors="raise")
    end_series = pd.to_datetime(end_times, errors="raise")
    mid_times = start_series + (end_series - start_series) / 2

    df = pd.DataFrame(
        {
            "window_start": start_series,
            "window_end": end_series,
            "window_mid": mid_times,
            "score_raw": scores_raw,
            "score_smoothed": scores_smoothed,
            "label": labels,
            "prediction": predictions,
            "adjusted_prediction": adjusted_predictions,
            "persistence_prediction": persistence_predictions,
            "persistence_adjusted_prediction": persistence_adjusted,
            "in_recalibration": in_recalibration,
        }
    )
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
