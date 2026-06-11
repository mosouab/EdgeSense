# Feedback-system verification runs

Captured end-to-end runs of the Layer-2 and Layer-3 verifiers against the live
simulation server (not the offline pipeline). Both scripts drive the running
device over WebSocket and compare identical labelled regions, so these numbers
are demonstrated, not just asserted.

Reproduce:

```
uv run python scripts/run_simulation.py        # terminal 1
uv run python scripts/verify_layer2.py         # terminal 2
uv run python scripts/verify_layer3.py
```

Run on 2026-06-11, Metro.PT, calibration 15 000 samples, speed 5000×.
Numbers vary slightly run-to-run (the dismissed-alarm sampling and the
benign/failure regions shift a little), but the direction is stable.

---

## Layer 2 — adapt (recalibrate on dismissed windows)

Dismiss a *benign* regime's false alarms → recalibrate (retrain the encoder on
calibration + dismissed windows) → re-score the SAME regions with the new model.

```
PASS 1 (baseline; dismiss benign-regime alarms)…
  [baseline] ROC=0.989 P=0.668 R=1.000 F1=0.801 thr=0.693 | benign-FP-rate=0.18 failure-recall=1.00
  dismissed 1 benign alarms; extra_healthy=6/59
RECALIBRATE…
  -> recalibrated, threshold 0.693 -> 0.672
PASS 2 (after recalibration, same regions)…
  [after L2] ROC=0.988 P=0.919 R=0.980 F1=0.948 thr=0.672 | benign-FP-rate=0.02 failure-recall=0.98

============================================================
LAYER-2 BEFORE / AFTER  (same regions, identical labels)
============================================================
  metric             baseline   after L2    delta
  roc                   0.989      0.988   -0.001
  precision             0.668      0.919   +0.251
  recall                1.000      0.980   -0.020
  f1                    0.801      0.948   +0.147
  benign_fp_rate        0.182      0.023   -0.159
  failure_recall        1.000      0.980   -0.020
```

Benign false-alarm rate 18% → 2%; failure recall 100% → 98% (held); precision
+25 pts; F1 +15 pts. Aggregate ROC stays flat — the win is at the operating
point, not in global ranking quality.

---

## Layer 3 — false-positive latent memory (instant, no retrain)

Dismiss a benign regime → its latent centroid is stored → re-visit the SAME
regime (no recalibration) and visit a real failure.

```
1) BENIGN regime (before any dismissal)…
   alert-rate=0.43  suppressed-rate=0.00
2) Dismiss the benign alarms (registers latent patterns; NO retrain)…
   dismissed 1 alarms -> 1 latent pattern(s) registered
3) RE-VISIT the same benign regime (no recalibration)…
   alert-rate=0.01  suppressed-rate=0.91
4) REAL failure region (measure on the FAULTY windows only)…

========================================================
LAYER-3 LATENT MEMORY  (instant, no retraining)
========================================================
  Benign regime   alert-rate : 0.43  ->  0.01   (suppressed 91%)
  Real failure    alert-rate : 1.00        (suppressed 0%)   [346 faulty windows]
```

The dismissed regime is silenced instantly (alert-rate 0.43 → 0.01, 91 %
suppressed) with no retraining, while the 346 faulty windows of a real failure
stay at alert-rate 1.00 with 0 % suppression. The score-cap guard is what keeps
the failure from being suppressed even where its latent is near the dismissed
centroid.
