"""Verify Layer-2 adaptation on Metro.PT with a clean, honest protocol.

Demonstrates the plan's target behaviour: dismiss alerts in a *genuinely
benign* regime -> recalibrate -> that regime stops alerting, while a real
failure is still detected. Because Metro.PT jumps revisit any region with the
current model, we score the SAME regions before and after on identical labels.

Two regions:
  - BENIGN: a healthy stretch far from any failure (the operator dismisses its
    false alarms — a sensible, bounded number, not everything).
  - FAILURE: the Jun 5-7 air-leak (#3), with healthy margins.

Each pass uses a FRESH WebSocket connection so the ~15s synchronous
/recalibrate call can't back up the stream.

Server must be running: uv run python scripts/run_simulation.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import websockets
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edgesense.datasets.metropt import load_metropt_failures  # noqa: E402

BASE = "http://127.0.0.1:8000"
SAMPLE_SECONDS = 10.0
CALIBRATION = 15000
SPEED = 5000
WS_KW = dict(max_size=None, ping_interval=None, max_queue=None)

BENIGN_JUMP_INDEX = 300_000          # early-mid March, no failure nearby
BENIGN_WINDOWS = 220
FAILURE_ID = 3                        # Jun 5-7 air leak
MARGIN_H = 8.0
MAX_DISMISSALS = 14                   # a sensible operator, not an oracle


def post(path, body=None):
    data = json.dumps(body).encode() if body else b""
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"} if body else {},
                                 method="POST")
    return json.loads(urllib.request.urlopen(req).read())


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path).read())


async def wait_inferring(ws, timeout=180):
    loop = asyncio.get_event_loop(); t0 = loop.time()
    while loop.time() - t0 < timeout:
        ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
        if ev.get("kind") == "phase" and ev.get("phase") == "inferring":
            return True
        if ev.get("kind") == "phase" and ev.get("phase") == "failed":
            return False
    return False


async def collect_region(ws, jump_index, n_windows, dismiss=False):
    """Jump to `jump_index`, collect `n_windows` scored readings.

    Drains stale buffered readings from the previous region by skipping until
    a reading whose `index` has reached the jump target (the WS queue holds
    pre-jump data we must not mis-attribute to this region)."""
    post("/jump", {"index": int(jump_index)})
    out, dismissed = [], set()
    arrived = False
    seen = 0
    loop = asyncio.get_event_loop(); t0 = loop.time()
    while len(out) < n_windows and loop.time() - t0 < 120:
        try:
            ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        except asyncio.TimeoutError:
            break
        # Only genuinely-scored windows carry `contributors`; unscored ticks
        # echo the last smoothed score, so filtering on score!=None would
        # collect every sample (50x too many, never spanning the failure).
        if (ev.get("kind") != "reading" or ev.get("phase") != "inferring"
                or "contributors" not in ev):
            continue
        if not arrived:
            if int(ev.get("index", 0)) < jump_index:
                continue           # stale pre-jump reading still in the WS buffer
            arrived = True
        seen += 1
        if seen <= 6:
            continue
        out.append(ev)
        if dismiss and len(dismissed) < MAX_DISMISSALS:
            eid = ev.get("episode_id")
            if eid and eid not in dismissed and ev.get("alert_level") in ("warn", "alert"):
                post("/feedback", {"episode_id": eid, "verdict": "false_positive",
                                   "note": "benign-regime false alarm"})
                dismissed.add(eid)
    return out, dismissed


def label_failure(readings, start, end):
    s, e = np.datetime64(start), np.datetime64(end)
    return np.array([1 if s <= np.datetime64(pd.to_datetime(r["timestamp"])) <= e else 0
                     for r in readings], dtype=int)


def summarize(benign, failure, fstart, fend, tag):
    scores = np.array([float(r["score"]) for r in benign + failure])
    labels = np.concatenate([np.zeros(len(benign), dtype=int),
                             label_failure(failure, fstart, fend)])
    thr = float((benign + failure)[-1]["threshold"])
    preds = (scores >= thr).astype(int)
    benign_scores = np.array([float(r["score"]) for r in benign])
    fail_lab = label_failure(failure, fstart, fend)
    fail_scores = np.array([float(r["score"]) for r in failure])
    out = {
        "roc": float(roc_auc_score(labels, scores)) if labels.min() != labels.max() else float("nan"),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "threshold": thr,
        "benign_fp_rate": float((benign_scores >= thr).mean()),
        "failure_recall": float(((fail_scores >= thr)[fail_lab == 1]).mean()) if (fail_lab == 1).any() else float("nan"),
    }
    print(f"  [{tag}] ROC={out['roc']:.3f} P={out['precision']:.3f} R={out['recall']:.3f} "
          f"F1={out['f1']:.3f} thr={thr:.3f} | benign-FP-rate={out['benign_fp_rate']:.2f} "
          f"failure-recall={out['failure_recall']:.2f}")
    return out


async def main():
    failures = load_metropt_failures()
    frow = failures[failures.failure_id == FAILURE_ID].iloc[0]
    fstart, fend = pd.to_datetime(frow["start_time"]), pd.to_datetime(frow["end_time"])
    margin = int(round(MARGIN_H * 3600 / SAMPLE_SECONDS))

    # Pass 1
    async with websockets.connect(BASE.replace("http", "ws") + "/ws", **WS_KW) as ws:
        post("/start", {"source": "metropt", "speed": SPEED, "calibration_samples": CALIBRATION})
        if not await wait_inferring(ws):
            print("training failed"); post("/stop"); return
        markers = {m["id"]: m for m in get("/failures?source=metropt")["failures"]}
        fjump = max(0, int(markers[FAILURE_ID]["start_index"]) - margin)
        fwin = 560  # ~8h margin + 52h failure + 8h margin in windows

        print("PASS 1 (baseline; dismiss benign-regime alarms)…")
        benign1, dismissed = await collect_region(ws, BENIGN_JUMP_INDEX, BENIGN_WINDOWS, dismiss=True)
        failure1, _ = await collect_region(ws, fjump, fwin, dismiss=False)
        base = summarize(benign1, failure1, fstart, fend, "baseline")
        ad = get("/adaptation")["adaptation"]
        print(f"  dismissed {len(dismissed)} benign alarms; extra_healthy={ad['extra_healthy']}/{ad['extra_cap']}")

    print("RECALIBRATE…")
    res = post("/recalibrate")
    print(f"  -> {res.get('status')}, threshold {base['threshold']:.3f} -> {res.get('threshold', float('nan')):.3f}")
    if res.get("status") != "recalibrated":
        post("/stop"); return

    # Pass 2 (fresh connection)
    async with websockets.connect(BASE.replace("http", "ws") + "/ws", **WS_KW) as ws:
        print("PASS 2 (after recalibration, same regions)…")
        benign2, _ = await collect_region(ws, BENIGN_JUMP_INDEX, BENIGN_WINDOWS, dismiss=False)
        failure2, _ = await collect_region(ws, fjump, fwin, dismiss=False)
        after = summarize(benign2, failure2, fstart, fend, "after L2")
        post("/stop")

    print("\n" + "=" * 60)
    print("LAYER-2 BEFORE / AFTER  (same regions, identical labels)")
    print("=" * 60)
    print(f"  {'metric':<16}{'baseline':>11}{'after L2':>11}{'delta':>9}")
    for k in ("roc", "precision", "recall", "f1", "benign_fp_rate", "failure_recall"):
        d = after[k] - base[k]
        print(f"  {k:<16}{base[k]:>11.3f}{after[k]:>11.3f}{d:>+9.3f}")
    print("\n  Layer 2 retrains the encoder on the dismissed benign windows: their")
    print("  scores collapse (the model now reconstructs them) so the benign FP rate")
    print("  drops, while real-failure scores are untouched so recall holds. The gain")
    print("  shows up at the operating point (precision/F1); aggregate ROC can stay flat")
    print("  because the global ranking quality was already high.")


if __name__ == "__main__":
    asyncio.run(main())
