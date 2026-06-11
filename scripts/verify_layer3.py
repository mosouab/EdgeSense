"""Verify Layer-3 false-positive latent memory (instant, no retraining).

Protocol on Metro.PT (jumps revisit any region with the current model):
  1. Visit a BENIGN regime; record its alert rate.
  2. Dismiss its alarms -> registers latent pattern(s). NO recalibration.
  3. Re-visit the SAME benign regime -> expect most windows suppressed and the
     alert rate to collapse.
  4. Visit a REAL failure -> expect NO suppression and the alert to still fire.

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from edgesense.datasets.metropt import load_metropt_failures  # noqa: E402

BASE = "http://127.0.0.1:8000"
SAMPLE_SECONDS = 10.0
CALIBRATION = 15000
SPEED = 5000
WS_KW = dict(max_size=None, ping_interval=None, max_queue=None)
BENIGN_JUMP = 300_000
BENIGN_WINDOWS = 220
FAILURE_ID = 3
MARGIN_H = 8.0
MAX_DISMISSALS = 14


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
        if ev.get("kind") == "phase" and ev.get("phase") in ("inferring", "failed"):
            return ev["phase"] == "inferring"
    return False


async def collect(ws, jump_index, n, dismiss=False):
    post("/jump", {"index": int(jump_index)})
    out, dismissed = [], set()
    arrived = False; seen = 0
    loop = asyncio.get_event_loop(); t0 = loop.time()
    while len(out) < n and loop.time() - t0 < 120:
        try:
            ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
        except asyncio.TimeoutError:
            break
        if ev.get("kind") != "reading" or ev.get("phase") != "inferring" or "contributors" not in ev:
            continue
        if not arrived:
            if int(ev.get("index", 0)) < jump_index:
                continue
            arrived = True
        seen += 1
        if seen <= 6:
            continue
        out.append(ev)
        if dismiss and len(dismissed) < MAX_DISMISSALS:
            eid = ev.get("episode_id")
            if eid and eid not in dismissed and ev.get("alert_level") in ("warn", "alert"):
                post("/feedback", {"episode_id": eid, "verdict": "false_positive", "note": "L3"})
                dismissed.add(eid)
    return out, dismissed


def rates(readings):
    n = max(1, len(readings))
    alert = sum(1 for r in readings if r.get("alert_level") in ("warn", "alert")) / n
    supp = sum(1 for r in readings if r.get("suppressed")) / n
    return alert, supp


async def main():
    failures = load_metropt_failures()
    frow = failures[failures.failure_id == FAILURE_ID].iloc[0]
    fstart, fend = pd.to_datetime(frow["start_time"]), pd.to_datetime(frow["end_time"])
    margin = int(round(MARGIN_H * 3600 / SAMPLE_SECONDS))

    async with websockets.connect(BASE.replace("http", "ws") + "/ws", **WS_KW) as ws:
        post("/start", {"source": "metropt", "speed": SPEED, "calibration_samples": CALIBRATION})
        if not await wait_inferring(ws):
            print("training failed"); post("/stop"); return
        markers = {m["id"]: m for m in get("/failures?source=metropt")["failures"]}
        fjump = max(0, int(markers[FAILURE_ID]["start_index"]) - margin)

        print("1) BENIGN regime (before any dismissal)…")
        b0, _ = await collect(ws, BENIGN_JUMP, BENIGN_WINDOWS, dismiss=False)
        a0, s0 = rates(b0)
        print(f"   alert-rate={a0:.2f}  suppressed-rate={s0:.2f}")

        print("2) Dismiss the benign alarms (registers latent patterns; NO retrain)…")
        b1, dismissed = await collect(ws, BENIGN_JUMP, BENIGN_WINDOWS, dismiss=True)
        pats = get("/patterns")["patterns"]
        print(f"   dismissed {len(dismissed)} alarms -> {len(pats)} latent pattern(s) registered")

        print("3) RE-VISIT the same benign regime (no recalibration)…")
        b2, _ = await collect(ws, BENIGN_JUMP, BENIGN_WINDOWS, dismiss=False)
        a2, s2 = rates(b2)
        print(f"   alert-rate={a2:.2f}  suppressed-rate={s2:.2f}")

        print("4) REAL failure region (measure on the FAULTY windows only)…")
        f1, _ = await collect(ws, fjump, 560, dismiss=False)
        post("/stop")

    # Restrict to windows whose timestamp is inside the labelled failure.
    s, e = np.datetime64(fstart), np.datetime64(fend)
    faulty = [r for r in f1 if s <= np.datetime64(pd.to_datetime(r["timestamp"])) <= e]
    fa, fs = rates(faulty) if faulty else (float("nan"), float("nan"))

    print("\n" + "=" * 56)
    print("LAYER-3 LATENT MEMORY  (instant, no retraining)")
    print("=" * 56)
    print(f"  Benign regime   alert-rate : {a0:.2f}  ->  {a2:.2f}   (suppressed {s2:.0%})")
    print(f"  Real failure    alert-rate : {fa:.2f}        (suppressed {fs:.0%})   "
          f"[{len(faulty)} faulty windows]")
    print("\n  Expect: benign alert-rate collapses with high suppression, while the")
    print("  faulty windows keep alerting and are NOT suppressed — all without a retrain.")


if __name__ == "__main__":
    asyncio.run(main())
