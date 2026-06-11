"""Produce a warm-start showcase model by driving the live HITL flow.

Cold-trains a source, then (for Metro.PT) performs the human-in-the-loop
adaptation automatically — dismiss a benign regime's false alarms, recalibrate
(Layer 2), then dismiss a second benign regime to seed Layer-3 latent patterns
against the recalibrated encoder — and saves the result as a warm-start model.

A later `/start` with `warm_model=<name>` loads it and infers immediately, with
the adaptation already baked in.

Server must be running: uv run python scripts/run_simulation.py
Then:                    uv run python scripts/pretrain_showcase.py --source metropt --name showcase
"""

from __future__ import annotations

import argparse
import asyncio
import json
import urllib.request

import websockets

BASE = "http://127.0.0.1:8000"
WS_KW = dict(max_size=None, ping_interval=None, max_queue=None)
# Two healthy Metro.PT stretches well before the first failure (~row 560k).
BENIGN_A = 300_000
BENIGN_B = 240_000
BENIGN_WINDOWS = 220
MAX_DISMISSALS = 14


def post(path, body=None):
    data = json.dumps(body).encode() if body else b""
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"} if body else {},
                                 method="POST")
    return json.loads(urllib.request.urlopen(req).read())


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path).read())


async def wait_inferring(ws, timeout=300):
    loop = asyncio.get_event_loop(); t0 = loop.time()
    while loop.time() - t0 < timeout:
        ev = json.loads(await asyncio.wait_for(ws.recv(), timeout=40))
        if ev.get("kind") == "phase":
            if ev.get("phase") == "inferring":
                return True
            if ev.get("phase") == "failed":
                return False
    return False


async def stream_and_dismiss(ws, jump_index, n_windows):
    """Jump to a benign region, dismiss its false alarms, return #dismissed."""
    post("/jump", {"index": int(jump_index)})
    dismissed = set()
    arrived = False; seen = 0; collected = 0
    loop = asyncio.get_event_loop(); t0 = loop.time()
    while collected < n_windows and loop.time() - t0 < 120:
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
        collected += 1
        if len(dismissed) < MAX_DISMISSALS:
            eid = ev.get("episode_id")
            if eid and eid not in dismissed and ev.get("alert_level") in ("warn", "alert"):
                post("/feedback", {"episode_id": eid, "verdict": "false_positive",
                                   "note": "showcase pretrain — benign regime"})
                dismissed.add(eid)
    return len(dismissed)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="metropt")
    ap.add_argument("--name", default="showcase")
    ap.add_argument("--calibration", type=int, default=30000)
    ap.add_argument("--speed", type=float, default=5000)
    ap.add_argument("--no-adapt", action="store_true", help="skip the HITL adaptation, just cold-train + save")
    args = ap.parse_args()

    async with websockets.connect(BASE.replace("http", "ws") + "/ws", **WS_KW) as ws:
        print(f"Cold-training {args.source} (calibration={args.calibration})…")
        post("/start", {"source": args.source, "speed": args.speed, "calibration_samples": args.calibration})
        if not await wait_inferring(ws):
            print("training failed"); post("/stop"); return

        if args.source == "metropt" and not args.no_adapt:
            print("HITL: dismiss benign regime A → recalibrate (Layer 2)…")
            d1 = await stream_and_dismiss(ws, BENIGN_A, BENIGN_WINDOWS)
            ad = get("/adaptation")["adaptation"]
            print(f"  dismissed {d1} alarms; extra_healthy={ad['extra_healthy']}/{ad['extra_cap']}")
            res = post("/recalibrate")
            print(f"  recalibrated → threshold {res.get('threshold'):.3f} (cleared {res.get('patterns_cleared',0)} stale patterns)")
            print("HITL: dismiss benign regime B → seed Layer-3 latent patterns…")
            d2 = await stream_and_dismiss(ws, BENIGN_B, BENIGN_WINDOWS)
            pats = get("/patterns")["patterns"]
            print(f"  dismissed {d2} alarms → {len(pats)} latent pattern(s) registered")

    res = post("/save_model", {"name": args.name, "include_calibration_windows": True})
    if res.get("status") != "saved":
        print(f"SAVE FAILED: {res}"); post("/stop"); return
    print(f"\nSaved warm-start model '{res['name']}' → {res['path']}")
    print(f"  adapted={res['adapted']}  n_patterns={res['n_patterns']}  "
          f"calibration_windows={res['has_calibration_windows']}")
    post("/stop")
    print("Done. Warm-start it from the UI (Model selector) or "
          f"POST /start {{source:'{args.source}', warm_model:'{args.name}'}}.")


if __name__ == "__main__":
    asyncio.run(main())
