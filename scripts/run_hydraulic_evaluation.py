"""Train and evaluate EdgeSense on the UCI Hydraulic Systems dataset.

We train one USAD model per fault component (cooler, valve, pump,
accumulator). Each model sees only cycles where its component is nominal
during training; it is then scored against all cycles, with positives
defined as "component degraded from nominal value".

This is a multi-fault story on one asset: the same architecture trained
with the same configuration adapts to four different fault modes.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from edgesense.datasets.hydraulic import (
    NOMINAL_VALUES,
    component_split,
    load_hydraulic_dataset,
)
from edgesense.models import USADConv1d, USADConv1dConfig
from edgesense.scoring import ScoringConfig, compute_usad_scores
from edgesense.training import (
    EarlyStoppingConfig,
    TrainingConfig,
    seed_all,
    split_train_validation,
    train_usad,
)

OUTPUT_DIR = Path("reports") / "hydraulic_evaluation"
COMPONENTS = ["cooler", "valve", "pump", "accumulator"]
HEALTHY_QUANTILE = 99.0


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("--- EdgeSense Hydraulic Multi-Fault Evaluation ---")
    print("[1/2] Loading and downsampling sensors...")
    dataset = load_hydraulic_dataset()
    print(
        f"  loaded {dataset.windows.shape[0]} cycles, "
        f"window shape per cycle = {dataset.windows.shape[1:]}, "
        f"sensors = {len(dataset.feature_columns)}"
    )

    results: dict[str, dict] = {}
    per_component_artifacts: dict[str, dict] = {}

    print("[2/2] Training and scoring per-component USAD...")
    for component in COMPONENTS:
        print(f"\n=== Component: {component} (nominal = {NOMINAL_VALUES[component]}) ===")
        split = component_split(dataset, component, train_fraction=0.7, seed=42)
        train_windows = dataset.windows[split["train_idx"]]
        calib_windows = dataset.windows[split["calib_idx"]]
        test_windows = dataset.windows[split["test_idx"]]
        test_labels = split["test_labels"].astype(np.int8)

        # Per-sensor standardization from the training set only.
        means = train_windows.mean(axis=(0, 1), keepdims=True)
        stds = train_windows.std(axis=(0, 1), keepdims=True) + 1e-6
        train_std = (train_windows - means) / stds
        calib_std = (calib_windows - means) / stds
        test_std = (test_windows - means) / stds

        # Inner train/val split for early stopping.
        train_only, val_only = split_train_validation(train_std, val_fraction=0.15)
        print(
            f"  train_only={train_only.shape[0]} val={val_only.shape[0]} "
            f"calib={calib_std.shape[0]} test={test_std.shape[0]} "
            f"(positives={int(test_labels.sum())})"
        )

        model_cfg = USADConv1dConfig(
            in_features=train_std.shape[2],
            base_channels=32,
            latent_channels=64,
            downsample_layers=2,
        )
        seed_all(42)
        model = USADConv1d(model_cfg)

        train_cfg = TrainingConfig(
            batch_size=64,
            epochs=80,
            learning_rate=1e-3,
            adv_ramp_epochs=30,
            adv_max_weight=0.3,
            grad_clip_norm=1.0,
        )
        stop_cfg = EarlyStoppingConfig(
            patience=12, min_delta=1e-4, max_epochs=80, val_fraction=0.15
        )
        history = train_usad(
            model,
            train_only,
            train_cfg,
            val_windows=val_only,
            early_stopping=stop_cfg,
            show_progress=False,
        )

        scoring_cfg = ScoringConfig(alpha=0.3, beta=0.7, batch_size=128)
        calib_scores = compute_usad_scores(model, calib_std, scoring_cfg)
        test_scores = compute_usad_scores(model, test_std, scoring_cfg)

        threshold = float(np.percentile(calib_scores, HEALTHY_QUANTILE))
        predictions = (test_scores >= threshold).astype(np.int8)

        # Metrics
        labels = test_labels
        block = {
            "n_train": int(train_only.shape[0]),
            "n_val": int(val_only.shape[0]),
            "n_calib": int(calib_std.shape[0]),
            "n_test": int(test_std.shape[0]),
            "n_test_positives": int(labels.sum()),
            "threshold": threshold,
            "best_val_recon_loss": (
                min(history.val_recon_losses) if history.val_recon_losses else None
            ),
            "epochs_trained": len(history.ae1_losses),
            "raw": {
                "precision": float(precision_score(labels, predictions, zero_division=0)),
                "recall": float(recall_score(labels, predictions, zero_division=0)),
                "f1": float(f1_score(labels, predictions, zero_division=0)),
                "accuracy": float(accuracy_score(labels, predictions)),
                "auc": (
                    float(roc_auc_score(labels, test_scores))
                    if labels.any() and (~labels.astype(bool)).any()
                    else None
                ),
            },
        }
        results[component] = block
        per_component_artifacts[component] = {
            "test_scores": test_scores,
            "test_labels": labels,
            "predictions": predictions,
            "threshold": threshold,
            "history": history,
        }
        print(
            f"  -> AUC {block['raw']['auc']:.3f}  "
            f"recall {block['raw']['recall']:.3f}  "
            f"precision {block['raw']['precision']:.3f}  "
            f"F1 {block['raw']['f1']:.3f}"
        )

    payload = {
        "components": results,
        "config": {
            "healthy_quantile": HEALTHY_QUANTILE,
            "train_fraction": 0.7,
            "window_seconds": 60,
            "sensors": dataset.feature_columns,
            "n_cycles": int(dataset.windows.shape[0]),
        },
    }
    with (OUTPUT_DIR / "metrics.json").open("w") as handle:
        json.dump(payload, handle, indent=2)

    # Save per-cycle scores so figures can pick them up later.
    rows = []
    for component, art in per_component_artifacts.items():
        for cycle_idx, (score, label, pred) in enumerate(
            zip(art["test_scores"], art["test_labels"], art["predictions"])
        ):
            rows.append(
                {
                    "component": component,
                    "cycle": cycle_idx,
                    "score": float(score),
                    "label": int(label),
                    "prediction": int(pred),
                    "threshold": art["threshold"],
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "per_cycle_scores.csv", index=False)

    print(f"\nSaved artifacts to {OUTPUT_DIR.resolve()}")
    print("\nSummary:")
    print(
        pd.DataFrame(
            {
                comp: {
                    "AUC": results[comp]["raw"]["auc"],
                    "Recall": results[comp]["raw"]["recall"],
                    "Precision": results[comp]["raw"]["precision"],
                    "F1": results[comp]["raw"]["f1"],
                }
                for comp in COMPONENTS
            }
        ).T.to_string(float_format=lambda x: f"{x:.3f}")
    )


if __name__ == "__main__":
    main()
