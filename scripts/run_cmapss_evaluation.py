"""Train and evaluate EdgeSense on NASA CMAPSS FD001.

Pipeline:
1. Window the per-unit cycle data into 30-cycle sliding windows.
2. Pretrain the USAD encoder unsupervised on windows that are still in the
   healthy regime (RUL clipped to MAX_RUL=125, i.e. the asset's normal
   operating envelope).
3. Freeze the encoder, train a small RUL regression head on all training
   windows with their piecewise-linear RUL targets.
4. Evaluate by predicting RUL on the last window of every test unit, then
   compare to RUL_FD001.txt ground truth. Report RMSE and the
   asymmetric CMAPSS score from Saxena et al. (2008).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

from edgesense.datasets.cmapss import (
    MAX_RUL,
    WINDOW_LENGTH,
    build_windows,
    cmapss_score,
    last_window_per_test_unit,
    load_cmapss_fd001,
)
from edgesense.models import RULHead, USADConv1d, USADConv1dConfig
from edgesense.training import (
    EarlyStoppingConfig,
    TrainingConfig,
    seed_all,
    split_train_validation,
    train_usad,
)

OUTPUT_DIR = Path("reports") / "cmapss_evaluation"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("--- EdgeSense CMAPSS FD001 RUL Evaluation ---")

    print("[1/4] Loading and windowing data...")
    dataset = load_cmapss_fd001()
    print(
        f"  train units: {len(dataset.train_units)}, "
        f"test units: {len(dataset.test_units)}, "
        f"features: {len(dataset.feature_columns)}"
    )

    train_windows, train_rul, train_units, _ = build_windows(
        dataset.train_units, dataset.feature_columns
    )
    # Re-derive normalization stats from the train set and reuse for test.
    raw_train = np.concatenate(
        [u[dataset.feature_columns].to_numpy(dtype=np.float32) for u in dataset.train_units.values()],
        axis=0,
    )
    means = raw_train.mean(axis=0)
    stds = raw_train.std(axis=0) + 1e-6

    test_windows, test_rul, test_units, _ = build_windows(
        dataset.test_units, dataset.feature_columns, means=means, stds=stds
    )
    final_windows, final_rul, final_unit_ids = last_window_per_test_unit(
        dataset.test_units, dataset.feature_columns, means=means, stds=stds
    )

    print(
        f"  train windows: {train_windows.shape[0]:,} | "
        f"test windows: {test_windows.shape[0]:,} | "
        f"final-window-per-test-unit: {final_windows.shape[0]}"
    )

    # Windows still in the "healthy" regime: target RUL == MAX_RUL.
    healthy_mask = train_rul >= MAX_RUL - 1e-3
    healthy_windows = train_windows[healthy_mask]
    print(
        f"  healthy windows (RUL = {MAX_RUL}): {healthy_windows.shape[0]:,} "
        f"({100 * healthy_mask.mean():.1f}% of train)"
    )

    print("[2/4] Pretraining USAD encoder on healthy windows...")
    model_cfg = USADConv1dConfig(
        in_features=train_windows.shape[2],
        base_channels=32,
        latent_channels=64,
        downsample_layers=2,
    )
    seed_all(42)
    usad = USADConv1d(model_cfg)

    train_only, val_only = split_train_validation(healthy_windows, val_fraction=0.1)
    pretrain_cfg = TrainingConfig(
        batch_size=128,
        epochs=60,
        learning_rate=1e-3,
        adv_ramp_epochs=25,
        adv_max_weight=0.3,
        grad_clip_norm=1.0,
    )
    stop_cfg = EarlyStoppingConfig(patience=10, min_delta=1e-4, max_epochs=60, val_fraction=0.1)
    pretrain_history = train_usad(
        usad,
        train_only,
        pretrain_cfg,
        val_windows=val_only,
        early_stopping=stop_cfg,
        show_progress=False,
    )
    print(
        f"  encoder trained for {len(pretrain_history.ae1_losses)} epochs, "
        f"best val_recon = {min(pretrain_history.val_recon_losses):.4f}"
    )

    print("[3/4] Training RUL head with frozen encoder...")
    rul_head = RULHead(latent_channels=model_cfg.latent_channels, hidden_dim=64, dropout=0.2)
    # Freeze USAD encoder.
    for p in usad.parameters():
        p.requires_grad = False
    usad.eval()

    train_idx_perm = np.random.default_rng(42).permutation(train_windows.shape[0])
    split = int(len(train_idx_perm) * 0.9)
    head_train_idx = train_idx_perm[:split]
    head_val_idx = train_idx_perm[split:]

    head_train_x = torch.tensor(train_windows[head_train_idx], dtype=torch.float32)
    head_train_y = torch.tensor(train_rul[head_train_idx], dtype=torch.float32)
    head_val_x = torch.tensor(train_windows[head_val_idx], dtype=torch.float32)
    head_val_y = torch.tensor(train_rul[head_val_idx], dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(head_train_x, head_train_y),
        batch_size=256,
        shuffle=True,
        drop_last=True,
    )
    optimizer = torch.optim.Adam(rul_head.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()
    rul_history: list[dict] = []
    best_val = float("inf")
    best_state: dict | None = None
    patience_counter = 0
    max_head_epochs = 80
    patience = 12

    for epoch in range(1, max_head_epochs + 1):
        rul_head.train()
        running = 0.0
        count = 0
        for batch_x, batch_y in loader:
            with torch.no_grad():
                latent = usad.encode(batch_x)
            preds = rul_head(latent)
            loss = criterion(preds, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rul_head.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
            count += 1
        train_mse = running / max(count, 1)

        rul_head.eval()
        with torch.no_grad():
            val_latent = usad.encode(head_val_x)
            val_preds = rul_head(val_latent)
            val_mse = float(((val_preds - head_val_y) ** 2).mean())
        rul_history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse})

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in rul_head.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"  early stopping at epoch {epoch} (best val MSE {best_val:.2f})")
            break

    if best_state is not None:
        rul_head.load_state_dict(best_state)
    rul_head.eval()

    print(
        f"  RUL head trained for {len(rul_history)} epochs, "
        f"best val MSE = {best_val:.2f} (RMSE = {best_val**0.5:.2f})"
    )

    print("[4/4] Evaluating on final-window per test unit...")
    with torch.no_grad():
        final_latent = usad.encode(torch.tensor(final_windows, dtype=torch.float32))
        final_preds = rul_head(final_latent).numpy()

    rmse = float(np.sqrt(((final_preds - final_rul) ** 2).mean()))
    mae = float(np.abs(final_preds - final_rul).mean())
    score = cmapss_score(final_preds, final_rul)

    # Pearson correlation between predicted and true RUL across the 100 test units.
    pearson = float(np.corrcoef(final_preds, final_rul)[0, 1])

    print(f"  Test-unit RMSE: {rmse:.2f}")
    print(f"  Test-unit MAE:  {mae:.2f}")
    print(f"  CMAPSS score:   {score:.1f}")
    print(f"  Pearson r:      {pearson:.3f}")

    # Reference: PHM 2008 winner-circle RMSE for FD001 is ~11-14
    # (Babu et al. 2016 CNN: 18.45; Zheng et al. 2017 LSTM: 16.14;
    # more recent transformers ~11-12). EdgeSense uses 1D-CNN encoder
    # without any domain-specific features.

    payload = {
        "rmse": rmse,
        "mae": mae,
        "cmapss_score": score,
        "pearson_r": pearson,
        "n_test_units": int(final_windows.shape[0]),
        "config": {
            "window_length": WINDOW_LENGTH,
            "max_rul_clip": MAX_RUL,
            "encoder_latent_channels": model_cfg.latent_channels,
            "features_used": dataset.feature_columns,
        },
        "pretrain": {
            "epochs": len(pretrain_history.ae1_losses),
            "best_val_recon": float(min(pretrain_history.val_recon_losses)),
        },
        "rul_head": {
            "epochs": len(rul_history),
            "best_val_mse": best_val,
        },
    }
    with (OUTPUT_DIR / "metrics.json").open("w") as f:
        json.dump(payload, f, indent=2)

    pd.DataFrame(
        {"unit": final_unit_ids, "true_rul": final_rul, "pred_rul": final_preds}
    ).to_csv(OUTPUT_DIR / "predictions_per_test_unit.csv", index=False)
    pd.DataFrame(rul_history).to_csv(OUTPUT_DIR / "rul_head_history.csv", index=False)

    print(f"\nSaved artifacts to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
