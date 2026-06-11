# EdgeSense — Human-in-the-Loop Feedback + Backlog Plan

Status: planning doc. Nothing here is built yet. Last updated 2026-06-11.

This plan turns operator feedback ("that alert was a false positive" / "yes,
that's a real fault") into (1) an audit trail, (2) alert dismissal, and
(3) on-asset model adaptation. It is built in three layers so each layer ships
and demos on its own. P2 is the prioritized backlog after that.

---

## Current-code facts this plan depends on

Verified 2026-06-11 (several earlier "gotchas" are now stale — see end):

- `device.py:_alert_level` is a hysteretic state machine that **only advances
  on a newly-scored window** (fixed). Episode detection should hook the
  *state transition* inside `_alert_level`, not per-sample events.
- During inference `_buffer` / `_cycle_buffer` are **trimmed to the last
  `window_length` rows** (fixed). Layer 2's episode-window capture therefore
  needs its **own bounded ring buffer** of recent scaled windows — it cannot
  read history back out of `_buffer`.
- `EdgeDevice.await_training()` exists and `SimulationState.stop()` awaits it,
  so training is no longer killable-but-orphaned; `stop()` blocks (bounded
  ~15 s) until the executor finishes. Design around *waiting*, not killing.
- Scoring weights: `ScoringConfig(alpha=0.3, beta=0.7)`. The deployed score is
  `0.3·MSE(x,AE1) + 0.7·MSE(x,AE2(AE1(x)))` (see `scoring.compute_usad_scores`).
- The CMMS feature (`src/edgesense/cmms/{base,builder,mock}.py`, `/work_request`
  endpoint, dashboard modal) is the **pattern to mirror** for feedback.
  `reports/work_orders/` is gitignored; do the same for `reports/feedback/`.
- Line numbers in the original spec have shifted after recent commits; search
  by symbol, not line.

---

## P1 — Human-in-the-loop feedback

### Layer 1 — Capture + acknowledge  (self-contained, ~half day)

**Goal:** operator can label an alert; we persist it and dismiss the alert.

**New file `src/edgesense/feedback.py`** (mirror `cmms/mock.py` shape):

```python
@dataclass
class FeedbackRecord:
    feedback_id: str            # FB-<utc-iso-compact>
    episode_id: str             # the alert episode being judged
    source: str                 # "metropt" | "hydraulic" | "cmapss"
    verdict: str                # "false_positive" | "confirmed"
    note: str
    created_at: str             # utc iso
    # episode snapshot (authoritative, taken server-side):
    started_at: str | None
    ended_at: str | None
    peak_score: float | None
    threshold: float | None
    contributors: list[dict]    # top contributors at peak
    diagnosis: dict | None
    forecast: dict | None
    def to_dict(self) -> dict: ...

class FeedbackStore:
    def __init__(self, root="reports/feedback"): ...
    def append(self, record: FeedbackRecord) -> Path:   # JSONL: <root>/<source>.jsonl
    def list(self, source: str | None = None) -> list[dict]:
```

Append-only JSONL, one file per source. Never mutate or delete rows.

**`device.py` changes:**

- **Episode tracking.** Add `self._episode: dict | None = None`. Inside
  `_alert_level`, on the transition into `"alert"`, start an episode:
  `{"episode_id": f"EP-{uuid4().hex[:10]}", "started_idx": ..., "peak_score": s,
  "peak_contributors": [...], "diagnosis": ..., "forecast": ...}`. While in
  alert, update `peak_score`/`peak_contributors` each scored window. On return
  to `"ok"`, finalize `ended_at` and keep the last finished episode in
  `self._last_episode` so feedback can reference it briefly after it clears.
  Expose `get_episode(episode_id)` returning the current or last-finished
  episode dict (so the API can snapshot it authoritatively).
- **`episode_id` in readings.** Add `episode_id` (current episode or None) to
  the `extra` of published readings during inference, so the UI knows which
  episode a feedback click refers to.
- **`force_release()`** — public method: `self._alert_state="ok"`,
  `self._above_streak = self._below_streak = self._warn_streak = 0`, finalize
  the current episode. Used when the operator dismisses a false positive so the
  pill returns to OK immediately rather than waiting out the release streak.

**`api.py` changes** (copy the `WorkRequestRequest` / `/work_request` pattern):

```python
class FeedbackRequest(BaseModel):
    episode_id: str
    verdict: str               # "false_positive" | "confirmed"
    note: str | None = None

@app.post("/feedback")
async def post_feedback(req: FeedbackRequest):
    # 1. snapshot episode authoritatively from state.device.get_episode(...)
    # 2. build + FeedbackStore().append(record)
    # 3. if verdict == false_positive: state.device.force_release()
    # 4. await state.bus.publish("ui.event", {"kind":"feedback","episode_id":...,
    #        "verdict":..., "feedback_id":...})
    # 5. return {"status":"recorded", **record.to_dict()}

@app.get("/feedback")
async def get_feedback(source: str | None = None):
    return {"feedback": FeedbackStore().list(source)}
```

**`dashboard.js` changes** (mirror the work-request modal):

- Two buttons next to "File work request" in the diagnosis action block:
  **"Confirm fault"** (verdict=confirmed) and **"False positive"**
  (verdict=false_positive). Only shown when an `episode_id` is present and the
  alert level is `warn`/`alert`.
- A small confirm popover (optional note field) → `POST /feedback`. Reuse the
  `workRequest` reactive sub-store shape (`state.feedback = { open, stage,
  loading, ... }`).
- In `handleEvent`, handle `kind === "feedback"`: flash a transient toast
  ("Recorded — alert dismissed" / "Confirmed"), and let the next reading's
  `alert_level` (now OK after `force_release`) repaint the pill.

**Verify Layer 1:** drive an alert (jump to a CMAPSS near-failure unit or a
Hydraulic fault), click "False positive" → pill returns to OK within one
flush, and `reports/feedback/<source>.jsonl` gains exactly one row with the
correct episode snapshot. Click "Confirm fault" on a separate episode →
JSONL row, pill unchanged.

---

### Layer 2 — Adapt  (the demo moment, ~1–1.5 days)

**Prereqs:** Layer 1 + alert-hysteresis fix (done) + a **scaled-window ring
buffer** (new) + **retained calibration windows** (new) + snapshot/versioning.

**Window ring buffer.** Add `self._recent_windows: collections.deque` (maxlen
~ a few hundred) of `(reading_index, scaled_window_np)` appended each scored
window during inference. This is the source of an episode's windows on FP.

**Retain calibration windows.** Training currently discards the `windows`
array. Keep `self._calibration_windows` (the scaled training windows) so a
retrain can concatenate extras. Memory note: Metro.PT 60k samples → ~1.2k
windows of (100,15) float32 ≈ 72 MB; acceptable on a demo laptop, but cap and
document. Cycle sources are far smaller.

**On false-positive feedback, collect extras:**

- Gather the dismissed episode's scaled windows from `_recent_windows` (those
  whose index falls in `[started_idx, ended_idx]`).
- Cap total injected "healthy" windows at **≤ 20% of calibration window count**
  (hard guard: an operator must not be able to teach a *real* degradation away
  by repeatedly dismissing it). Track cumulative injected count; refuse beyond
  the cap and surface a UI message.
- Stash as `self._extra_healthy_windows` (append across multiple FPs, subject
  to the cap).

**`POST /recalibrate`:**

- Reuse the existing pause-source → train-in-executor lifecycle
  (`await_training` already makes teardown safe).
- Retrain on `concat(self._calibration_windows, self._extra_healthy_windows)`,
  recompute the p99 threshold from the new healthy scores, rebuild baseline
  contributions.
- Publish phase events so the UI shows "recalibrating…".

**Versioning + revert:**

- On every (re)train, write a snapshot to
  `reports/feedback/snapshots/<snapshot_id>/`:
  `model.pt` (state_dict), `meta.json` (threshold, healthy_quantile,
  scaler params, feedback_ids included, parent_snapshot_id, created_at).
- `POST /revert` restores the previous snapshot (model + threshold + scaler)
  without touching the append-only feedback log.
- Keep a small in-memory stack of snapshot ids for the UI revert button.

**Verify Layer 2:** start a source, let it alert on a *benign regime* (e.g.
Hydraulic idle / a known-noisy Metro.PT plateau), dismiss → recalibrate → the
**same segment no longer alerts**, but jumping to a **real labelled failure
still alerts**. Then `POST /revert` and confirm the benign segment alerts again
(proving versioning works).

---

### Layer 3 — FP latent memory  (stretch, ~1 day)

No retraining. On dismissal, store the **encoder-latent centroid** of the
dismissed episode (`model.encode(window).mean(dim=2)`, averaged over the
episode's windows) plus a radius (e.g. mean + k·std of intra-episode distance).

At inference, compute the current window's latent distance to each stored
centroid; if within radius, **downgrade alert→warn** tagged
`"matches dismissed pattern #N"` in the diagnosis. Per-pattern revert = delete
one centroid. Persist centroids alongside feedback snapshots.

Advantage over Layer 2: instant (no retrain), reversible per-pattern, and it
*explains* the suppression ("this looks like the thing you dismissed on
Tuesday") rather than silently absorbing it into the model.

---

## P2 — Backlog (priority order)

1. **Edge footprint panel.** Per-window inference latency
   (`time.perf_counter` around `compute_usad_scores`), model size (checkpoint
   bytes ≈ 173 KB / param count × 4), process RSS (`psutil`). Publish in a
   periodic stat event; render a small panel. Optional ONNX export + benchmark.
   *This is the only evidence behind the "edge-native" claim — highest-value
   backlog item for the pitch.*
2. **Warm start.** Cache `(scaler, model state_dict, threshold)` keyed by
   `(source, calibration_samples, seed)` to disk; `/start` checks the cache and
   skips straight to inference. Invalidate on code/version change. Demo
   insurance against the 3-min Metro.PT calibration.
3. **README "Live demo" section + screenshot.** The console is invisible to
   repo readers; one screenshot + a paragraph closes that gap.
4. **Fleet view.** Namespaced bus topics `ui.event.<asset>`, multiple devices,
   a card-per-asset landing page. Larger architecture change — defer until P1
   lands.
5. **Drift monitor.** KS / PSI of the current feature distribution vs the
   healthy reference → a "recalibration recommended" banner. Ties directly to
   the 24%→57% recalibration result in the README.
6. **Expected-vs-actual reconstruction overlay.** `_compute_feature_
   contributions` already computes `recon1` and throws it away; expose it and
   plot input vs reconstruction for the worst channel — strong explainability
   visual.
7. **Smoke tests (none exist).** pytest: (a) app boots + `GET /` returns 200;
   (b) tiny synthetic train→score round-trip through `EdgeDevice`/training.
   Cheap insurance against regressions like the ones just fixed.
8. **Align early-stop criterion with the deployed score.**
   `training.py:_evaluate_reconstruction_loss` validates on AE1-only MSE, but
   deployment scores `0.3·AE1 + 0.7·AE2`. ⚠️ **Changes the offline model and
   the published metrics (88/57/0.69)** — must re-run `run_full_evaluation.py`,
   regenerate figures, and update the README in the *same* commit. Do not do
   piecemeal.
9. **Hydraulic valve fix.** Preserve the 100 Hz pressure transients instead of
   downsampling them away; turns valve AUC 0.54 into a per-asset-adaptation
   story.

---

## Gotchas (updated for current code)

- **`SourceSpec.feature_names` is mutated in place** after dataset load and the
  device holds the list reference — never replace the list object, mutate it.
- **`stop()` now awaits in-flight training** (`await_training`); it can block
  ~15 s but will not orphan an executor. Design around waiting, not killing.
- **Sim ≠ offline eval.** Two pipelines drift (in-sample vs held-out threshold,
  different smoothing kernels). If you touch one, check the other. The cleanest
  long-term fix is one shared `calibrate()` — a deliberate refactor that will
  move the published numbers, so gate it behind a metrics re-run.
- **Inference buffers are trimmed to `window_length`** — Layer 2's episode
  capture needs its own ring buffer (above).
- **Hydraulic / CMAPSS download on first use** (~80 / 10 MB); Metro.PT CSV
  ships in `data/` (1.5 M rows, no NaNs).
- `reports/feedback/` and `reports/feedback/snapshots/` must be gitignored
  (runtime artefacts), like `reports/work_orders/`.
- *(stale, now fixed: the `train_usad` docstring schedule, the alert-hysteresis
  bug, and stop()-cannot-stop-training are all resolved as of 2026-06-11.)*

---

## Suggested sequencing

1. Layer 1 end-to-end (capture + dismiss) — ships value, zero offline risk.
2. P2 #7 smoke tests — lock in current behaviour before deeper changes.
3. Layer 2 (ring buffer → retain windows → /recalibrate → snapshot/revert).
4. P2 #1 edge footprint panel (independent, high pitch value).
5. Layer 3 latent memory (stretch).
6. P2 #8 (offline-affecting) only as a deliberate, metrics-reconciled commit.
