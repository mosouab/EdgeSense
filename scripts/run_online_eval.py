"""Drive the LIVE streaming simulation end-to-end and score it.

Unlike scripts/run_full_evaluation.py (the offline pipeline), this connects to
the running sim server over WebSocket, runs one source from calibration through
to stream end, and computes precision / recall / ROC-AUC against the per-cycle
ground truth the source streams.

It also flags every false positive and shows what operator feedback-driven
threshold recalibration would do to the numbers (the cheap part of Layer 2):
when the operator dismisses the false-positive alerts, those windows join the
healthy reference and the p99 threshold is refit.

Hydraulic is the default because it streams a clean per-cycle binary label
(`true_anomaly`); Metro.PT / CMAPSS don't stream window labels.

Usage (server must be running: `uv run python scripts/run_simulation.py`):
    uv run python scripts/run_online_eval.py --source hydraulic --speed 5000 --calibration 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import urllib.request

import numpy as np
import websockets
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE = "http://127.0.0.1:8000"


def post(path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body else b""
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST",
    )
    return json.loads(urllib.request.urlopen(req).read())


async def drive(source: str, speed: float, calibration: int, timeout: float = 240.0) -> list[dict]:
    """Run one source to completion; return the list of inference readings."""

    readings: list[dict] = []
    async with websockets.connect(BASE.replace("http", "ws") + "/ws", max_size=None) as ws:
        post("/start", {"source": source, "speed": speed, "calibration_samples": calibration})
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        phase = None
        while loop.time() - t0 < timeout:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5)
            except asyncio.TimeoutError:
                # If we were already inferring and the stream went quiet, assume done.
                if phase in ("inferring", "finished"):
                    break
                continue
            ev = json.loads(msg)
            if ev.get("kind") == "phase":
                phase = ev.get("phase")
                if phase in ("finished", "failed"):
                    break
            elif ev.get("kind") == "reading" and ev.get("phase") == "inferring":
                if ev.get("score") is not None and ev.get("true_anomaly") is not None:
                    readings.append(ev)
        post("/stop")
    return readings


def binary_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    return {
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="hydraulic")
    ap.add_argument("--speed", type=float, default=5000)
    ap.add_argument("--calibration", type=int, default=200)
    ap.add_argument("--max-fp", type=int, default=12, help="max FP episodes to print")
    args = ap.parse_args()

    print(f"Driving live sim: source={args.source} speed={args.speed} calibration={args.calibration}")
    readings = asyncio.run(drive(args.source, args.speed, args.calibration))
    if not readings:
        print("No labelled inference readings collected — is the server running and does "
              "this source stream `true_anomaly`? (Hydraulic does; Metro.PT/CMAPSS don't.)")
        return

    labels = np.array([int(r["true_anomaly"]) for r in readings], dtype=int)
    scores = np.array([float(r["score"]) for r in readings], dtype=float)
    thr = float(readings[-1]["threshold"])  # device's calibration p99 threshold

    n = len(readings)
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    print(f"\nInference cycles scored: {n}  (faulty={n_pos}, nominal={n_neg})")
    print(f"Device threshold (p99 of calibration healthy): {thr:.4f}")

    if n_pos == 0 or n_neg == 0:
        print("Only one class present — cannot compute ROC/precision/recall. "
              "Increase the run length or lower calibration.")
        return

    roc = float(roc_auc_score(labels, scores))

    # ── Baseline: predict with the device threshold ──────────────────────
    preds = (scores >= thr).astype(int)
    base = binary_metrics(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    print("\n" + "=" * 64)
    print("BASELINE  (device p99 threshold, no feedback)")
    print("=" * 64)
    print(f"  ROC-AUC   : {roc:.3f}")
    print(f"  Precision : {base['precision']:.3f}")
    print(f"  Recall    : {base['recall']:.3f}")
    print(f"  F1        : {base['f1']:.3f}")
    print(f"  Confusion : TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # ── Flag every false positive, grouped into contiguous episodes ──────
    fp_idx = np.where((preds == 1) & (labels == 0))[0]
    print("\n" + "-" * 64)
    print(f"FALSE POSITIVES: {len(fp_idx)} cycles flagged the operator would dismiss")
    print("-" * 64)
    episodes = _group_episodes(fp_idx)
    print(f"  ({len(episodes)} contiguous FP episodes; showing top {args.max_fp} by peak score)")
    ranked = sorted(
        episodes,
        key=lambda ab: float(np.max(scores[ab[0]: ab[1] + 1])),
        reverse=True,
    )
    for k, (a, b) in enumerate(ranked[: args.max_fp], 1):
        seg = slice(a, b + 1)
        peak_i = a + int(np.argmax(scores[seg]))
        top = _top_contributor(readings[peak_i])
        print(f"  FP #{k}: cycles {a}-{b}  ({b - a + 1} cyc)  "
              f"peak {scores[peak_i]:.3f} (thr {thr:.3f})  driver: {top}")

    # ── Feedback-driven threshold recalibration ──────────────────────────
    # The operator dismisses the FP alerts, so those windows + the confirmed
    # nominal windows become the healthy reference. Refit p99 over them.
    healthy_scores = scores[labels == 0]
    new_thr = float(np.percentile(healthy_scores, 99.0))
    preds2 = (scores >= new_thr).astype(int)
    after = binary_metrics(labels, preds2)
    tn2, fp2, fn2, tp2 = confusion_matrix(labels, preds2, labels=[0, 1]).ravel()

    print("\n" + "=" * 64)
    print("AFTER FEEDBACK  (p99 threshold refit over operator-confirmed healthy)")
    print("=" * 64)
    print(f"  New threshold : {new_thr:.4f}  (was {thr:.4f})")
    print(f"  ROC-AUC       : {roc:.3f}   (UNCHANGED — ROC ranks the scores; "
          f"only retraining (Layer 2) moves it)")
    print(f"  Precision     : {after['precision']:.3f}   (was {base['precision']:.3f})")
    print(f"  Recall        : {after['recall']:.3f}   (was {base['recall']:.3f})")
    print(f"  F1            : {after['f1']:.3f}   (was {base['f1']:.3f})")
    print(f"  Confusion     : TP={tp2}  FP={fp2}  FN={fn2}  TN={tn2}")

    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  {'metric':<12}{'baseline':>12}{'after fb':>12}{'delta':>10}")
    for key, label in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]:
        d = after[key] - base[key]
        print(f"  {label:<12}{base[key]:>12.3f}{after[key]:>12.3f}{d:>+10.3f}")
    print(f"  {'ROC-AUC':<12}{roc:>12.3f}{roc:>12.3f}{0.0:>+10.3f}")
    print(f"\n  False positives: {fp} -> {fp2}  ({fp - fp2:+d})")
    print("  Note: Layer-1 feedback / threshold recalibration trades precision for")
    print("  recall along the same ROC curve. Moving ROC itself needs Layer-2 retraining")
    print("  (inject the dismissed windows as healthy and refit the encoder).")


def _group_episodes(indices: np.ndarray) -> list[tuple[int, int]]:
    """Group sorted indices into contiguous [start, end] runs."""
    if len(indices) == 0:
        return []
    episodes = []
    start = prev = int(indices[0])
    for i in indices[1:]:
        i = int(i)
        if i == prev + 1:
            prev = i
        else:
            episodes.append((start, prev))
            start = prev = i
    episodes.append((start, prev))
    return episodes


def _top_contributor(reading: dict) -> str:
    contribs = reading.get("contributors") or []
    if not contribs:
        return "—"
    c = max(contribs, key=lambda x: x.get("delta_pct", 0.0))
    return f"{c.get('label', c.get('name', '?'))} (+{c.get('delta_pct', 0):.1f} pts)"


if __name__ == "__main__":
    main()
