"""The simulated edge device.

Receives sensor events from a `DataSource` via the event bus, buffers
windows during a calibration phase, trains a USAD model in a background
thread once enough healthy data is collected, then runs streaming
inference and publishes anomaly score / health score / alert events.

The device is deliberately stateless about WHICH dataset it's running
on; it just consumes the spec from the source.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from ..evaluation import apply_median_filter
from ..health import health_score
from ..models import USADConv1d, USADConv1dConfig
from ..scoring import ScoringConfig, compute_usad_scores
from ..training import EarlyStoppingConfig, TrainingConfig, seed_all, split_train_validation, train_usad
from .bus import EventBus
from .source import DataSource, SensorEvent

# Minimum number of sliding windows the calibration buffer must yield before
# we attempt to train. A 90/10 train/val split needs enough on each side for
# early stopping to be meaningful.
_MIN_CALIBRATION_WINDOWS = 40


@dataclass
class DeviceConfig:
    calibration_samples: int = 30_000
    healthy_quantile: float = 99.0
    scoring_alpha: float = 0.3
    scoring_beta: float = 0.7
    median_smoothing_window: int = 21
    base_channels: int = 32
    latent_channels: int = 64
    downsample_layers: int = 2
    batch_size: int = 256
    learning_rate: float = 1e-3
    max_epochs: int = 25
    adv_ramp_epochs: int = 15
    adv_max_weight: float = 0.3
    seed: int = 42
    # Alert hysteresis: how many consecutive windows must agree before the
    # alert state flips, and the release threshold (fraction of the trigger).
    alert_trigger_streak: int = 4
    alert_release_streak: int = 6
    alert_release_fraction: float = 0.65
    # Trend forecast: rolling buffer of recent (asset_seconds, smoothed_score)
    # samples used to fit a linear regression and extrapolate when the score
    # will cross the threshold.
    forecast_buffer_size: int = 200
    forecast_min_samples: int = 30
    forecast_significance_t_stat: float = 2.0


@dataclass
class DevicePhase:
    """Snapshot of the device's current lifecycle state, broadcast to UI."""

    name: str
    progress: float
    detail: str = ""


class EdgeDevice:
    """Run the calibration -> train -> infer lifecycle for one DataSource."""

    def __init__(
        self,
        bus: EventBus,
        source: DataSource,
        cfg: DeviceConfig,
        pause_event: asyncio.Event | None = None,
    ) -> None:
        self.bus = bus
        self.source = source
        self.cfg = cfg
        # Optional asyncio.Event the device sets while training so the source
        # pauses and doesn't exhaust its sequence before inference can begin.
        self._pause_event = pause_event
        self._buffer: list[dict[str, float]] = []
        self._cycle_buffer: list[np.ndarray] = []
        self._scaler: StandardScaler | None = None
        self._model: USADConv1d | None = None
        self._threshold: float | None = None
        self._healthy_reference: np.ndarray | None = None
        self._rolling_scores: list[float] = []
        self._rolling_contributions: list[np.ndarray] = []
        self._baseline_contributions: np.ndarray | None = None
        # Trend forecast buffer: (asset_seconds, smoothed_score) tuples.
        self._forecast_buffer: list[tuple[float, float]] = []
        self._last_forecast: dict[str, Any] | None = None
        # Cached diagnostic ticket so non-scored events can still publish one.
        self._last_diagnosis: dict[str, Any] | None = None
        self._phase = DevicePhase(name="awaiting", progress=0.0)
        # Samples seen since the last scored window (sample-based sources).
        self._stride_counter = 0
        self._training_task: asyncio.Task | None = None
        # Sticky alert state with hysteresis: only flip after N consecutive
        # windows agree, with separate trigger and release thresholds.
        self._alert_state: str = "ok"
        self._above_streak: int = 0
        self._below_streak: int = 0
        self._warn_streak: int = 0
        # For cycle sources with discrete assets (CMAPSS units): the asset id
        # currently being streamed. When it changes during inference we reset
        # the cycle buffer so windows never blend two different engines.
        self._current_unit_id: Any = None

    @property
    def phase(self) -> DevicePhase:
        return self._phase

    async def await_training(self, timeout: float = 30.0) -> None:
        """Block until any in-flight training executor finishes.

        The torch training runs in a thread-pool executor that cannot be
        cancelled mid-flight; awaiting it here lets a caller (e.g. stop())
        guarantee a clean teardown before a new simulation starts, so two
        trainings never run concurrently.
        """

        task = self._training_task
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
        except Exception:
            pass

    async def run(self, source_stream) -> None:
        """Consume the source stream end-to-end."""

        spec = self.source.spec
        cycle_based = spec.cycle_based
        feature_names = spec.feature_names
        window_length = spec.window_length
        stride = spec.stride
        calibration_target = self.cfg.calibration_samples
        unit_label = "cycles" if cycle_based else "samples"
        await self._broadcast_phase(
            "calibrating", 0.0, f"collecting {calibration_target:,} {unit_label}"
        )

        async for event in source_stream:
            if event.metadata.get("jumped"):
                self._buffer.clear()
                self._cycle_buffer.clear()
                self._rolling_scores = []
                self._rolling_contributions = []
                self._forecast_buffer = []
                self._last_forecast = None
                self._last_diagnosis = None
                self._stride_counter = 0
                self._alert_state = "ok"
                self._above_streak = 0
                self._below_streak = 0
                self._warn_streak = 0
                self._current_unit_id = None
                await self._broadcast_phase(
                    self._phase.name,
                    self._phase.progress,
                    f"jumped to row {event.index} ({event.timestamp})",
                )

            if cycle_based:
                await self._handle_cycle_event(
                    event, feature_names, window_length, calibration_target
                )
            else:
                await self._handle_sample_event(
                    event, feature_names, window_length, stride, calibration_target
                )

        if self._training_task is not None:
            await self._training_task
        await self._broadcast_phase("finished", 1.0, "stream complete")

    async def _handle_sample_event(
        self,
        event: SensorEvent,
        feature_names: list[str],
        window_length: int,
        stride: int,
        calibration_target: int,
    ) -> None:
        self._buffer.append(event.features)
        phase_name = self._phase.name

        if phase_name == "calibrating":
            progress = min(len(self._buffer) / max(calibration_target, 1), 1.0)
            if len(self._buffer) % 200 == 0:
                await self._broadcast_phase(
                    "calibrating",
                    progress,
                    f"{len(self._buffer):,} / {calibration_target:,} samples",
                )
            await self._publish_reading(event, score=None, health=100.0, alert_level="ok", phase="calibrating")
            if len(self._buffer) >= calibration_target and self._training_task is None:
                await self._broadcast_phase("training", 0.0, "fitting scaler + USAD model")
                # Snapshot the calibration buffer so the executor thread trains
                # on a stable copy while the event loop keeps appending.
                buffer_snapshot = list(self._buffer)
                self._training_task = asyncio.create_task(
                    self._train_async(buffer_snapshot, feature_names, window_length, stride)
                )

        elif phase_name == "training":
            await self._publish_reading(event, score=None, health=100.0, alert_level="ok", phase="training")

        elif phase_name == "inferring":
            # Keep the buffer bounded — only the last `window_length` samples
            # are ever needed to score the current window.
            if len(self._buffer) > window_length:
                del self._buffer[:-window_length]
            self._stride_counter += 1
            if len(self._buffer) >= window_length and self._stride_counter >= stride:
                self._stride_counter = 0
                score, smoothed, contributors = await self._score_window(
                    feature_names, window_length
                )
                health = float(
                    health_score(
                        np.asarray([smoothed], dtype=np.float32),
                        self._healthy_reference,
                        self._threshold,
                    )[0]
                )
                alert_level = self._alert_level(smoothed, self._threshold)
                forecast = self._update_forecast(event, smoothed)
                diagnosis = self._build_diagnosis(alert_level, forecast, contributors)
                self._last_diagnosis = diagnosis
                await self._publish_reading(
                    event,
                    score=smoothed,
                    health=health,
                    alert_level=alert_level,
                    phase="inferring",
                    extra={
                        "contributors": contributors,
                        "forecast": forecast,
                        "diagnosis": diagnosis,
                    },
                )
            else:
                last_smoothed = self._rolling_scores[-1] if self._rolling_scores else None
                last_health = (
                    float(
                        health_score(
                            np.asarray([last_smoothed], dtype=np.float32),
                            self._healthy_reference,
                            self._threshold,
                        )[0]
                    )
                    if last_smoothed is not None and self._threshold is not None
                    else 100.0
                )
                # No new window was scored this tick, so DON'T advance the
                # hysteresis state machine — just echo the sticky state.
                # Advancing here would let `stride` repeated samples satisfy
                # the "N consecutive windows" trigger from a single window.
                alert_level = self._alert_state
                extras: dict[str, Any] = {}
                if self._last_forecast:
                    extras["forecast"] = self._last_forecast
                if self._last_diagnosis:
                    extras["diagnosis"] = self._last_diagnosis
                await self._publish_reading(
                    event,
                    score=last_smoothed,
                    health=last_health,
                    alert_level=alert_level,
                    phase="inferring",
                    extra=extras or None,
                )

    async def _handle_cycle_event(
        self,
        event: SensorEvent,
        feature_names: list[str],
        window_length: int,
        calibration_target: int,
    ) -> None:
        if event.cycle_features is None:
            raise ValueError(
                "Cycle-based source emitted an event without cycle_features."
            )

        # During inference, a change of asset id (e.g. CMAPSS test engine #1 ->
        # #2) means the buffered tail belongs to a different machine. Clear the
        # window buffer and per-asset state so we never score a window that
        # blends two engines, and so the trend forecast restarts per asset.
        # During calibration we deliberately accumulate all train units.
        unit_id = event.metadata.get("unit_id")
        if (
            self._phase.name == "inferring"
            and unit_id is not None
            and unit_id != self._current_unit_id
        ):
            self._cycle_buffer.clear()
            self._rolling_scores = []
            self._rolling_contributions = []
            self._forecast_buffer = []
            self._last_forecast = None
        if unit_id is not None:
            self._current_unit_id = unit_id

        self._cycle_buffer.append(event.cycle_features)
        count = len(self._cycle_buffer)
        phase_name = self._phase.name

        if phase_name == "calibrating":
            progress = min(count / max(calibration_target, 1), 1.0)
            if count % 25 == 0 or count == calibration_target:
                await self._broadcast_phase(
                    "calibrating",
                    progress,
                    f"{count:,} / {calibration_target:,} cycles",
                )
            await self._publish_reading(event, score=None, health=100.0, alert_level="ok", phase="calibrating")
            if count >= calibration_target and self._training_task is None:
                await self._broadcast_phase("training", 0.0, "fitting scaler + USAD model on cycles")
                cycles_snapshot = list(self._cycle_buffer)
                self._training_task = asyncio.create_task(
                    self._train_async_cycles(cycles_snapshot, window_length)
                )

        elif phase_name == "training":
            await self._publish_reading(event, score=None, health=100.0, alert_level="ok", phase="training")

        elif phase_name == "inferring":
            # Keep the cycle buffer bounded — at most `window_length` cycles
            # are needed to assemble the current window.
            if len(self._cycle_buffer) > window_length:
                del self._cycle_buffer[:-window_length]
            window = self._extract_cycle_window(self._cycle_buffer, window_length)
            if window is None or self._scaler is None or self._model is None:
                await self._publish_reading(event, score=None, health=100.0, alert_level="ok", phase="inferring")
                return
            score, smoothed, contributors = await self._score_cycle_window(window, feature_names)
            health = float(
                health_score(
                    np.asarray([smoothed], dtype=np.float32),
                    self._healthy_reference,
                    self._threshold,
                )[0]
            )
            alert_level = self._alert_level(smoothed, self._threshold)
            forecast = self._update_forecast(event, smoothed)
            diagnosis = self._build_diagnosis(alert_level, forecast, contributors)
            self._last_diagnosis = diagnosis
            await self._publish_reading(
                event,
                score=smoothed,
                health=health,
                alert_level=alert_level,
                phase="inferring",
                extra={
                    "true_anomaly": event.metadata.get("is_anomaly"),
                    "unit_id": event.metadata.get("unit_id"),
                    "unit_cycle": event.metadata.get("unit_cycle"),
                    "contributors": contributors,
                    "forecast": forecast,
                    "diagnosis": diagnosis,
                },
            )

    async def _train_async(
        self,
        buffer_snapshot: list[dict[str, float]],
        feature_names: list[str],
        window_length: int,
        stride: int,
    ) -> None:
        if self._pause_event is not None:
            self._pause_event.set()
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                self._train_blocking,
                buffer_snapshot,
                list(feature_names),
                window_length,
                stride,
            )
        except Exception as exc:
            if self._pause_event is not None:
                self._pause_event.clear()
            await self._broadcast_phase("failed", 0.0, f"training error: {exc}")
            return
        if self._pause_event is not None:
            self._pause_event.clear()
        await self._broadcast_phase(
            "inferring", 1.0, f"threshold = {self._threshold:.3f}"
        )

    def _train_blocking(
        self,
        buffer_snapshot: list[dict[str, float]],
        feature_names: list[str],
        window_length: int,
        stride: int,
    ) -> None:
        """Heavy lifting: fit scaler, build windows, train USAD, set threshold.

        Operates on a snapshot of the calibration buffer (not self._buffer)
        so the live event loop can keep appending without racing this thread.
        """

        df = np.asarray(
            [[row[name] for name in feature_names] for row in buffer_snapshot],
            dtype=np.float32,
        )
        scaler = StandardScaler().fit(df)
        scaled = scaler.transform(df).astype(np.float32)

        # Sliding windows over the calibration buffer.
        num_windows = max(0, (len(scaled) - window_length) // stride + 1)
        if num_windows < _MIN_CALIBRATION_WINDOWS:
            raise ValueError(
                f"Not enough calibration data: {len(scaled)} samples produced "
                f"{num_windows} windows of length {window_length} at stride "
                f"{stride}; need at least {_MIN_CALIBRATION_WINDOWS}. Increase "
                f"the calibration sample count."
            )
        windows = np.stack(
            [scaled[i * stride : i * stride + window_length] for i in range(num_windows)],
            axis=0,
        )

        seed_all(self.cfg.seed)
        cfg_model = USADConv1dConfig(
            in_features=scaled.shape[1],
            base_channels=self.cfg.base_channels,
            latent_channels=self.cfg.latent_channels,
            downsample_layers=self.cfg.downsample_layers,
        )
        model = USADConv1d(cfg_model)

        train_only, val_only = split_train_validation(windows, val_fraction=0.1)
        train_cfg = TrainingConfig(
            batch_size=min(self.cfg.batch_size, train_only.shape[0]),
            epochs=self.cfg.max_epochs,
            learning_rate=self.cfg.learning_rate,
            adv_ramp_epochs=self.cfg.adv_ramp_epochs,
            adv_max_weight=self.cfg.adv_max_weight,
            grad_clip_norm=1.0,
            seed=self.cfg.seed,
        )
        stop_cfg = EarlyStoppingConfig(patience=6, min_delta=1e-4, max_epochs=self.cfg.max_epochs, val_fraction=0.1)
        train_usad(model, train_only, train_cfg, val_windows=val_only, early_stopping=stop_cfg, show_progress=False)

        scoring_cfg = ScoringConfig(alpha=self.cfg.scoring_alpha, beta=self.cfg.scoring_beta, batch_size=256)
        cal_scores = compute_usad_scores(model, windows, scoring_cfg, show_progress=False)
        # Median-smooth at the configured kernel length.
        kernel = self.cfg.median_smoothing_window
        if kernel >= 3 and kernel % 2 == 1 and cal_scores.size:
            smoothed = apply_median_filter(cal_scores, kernel).astype(np.float32)
        else:
            smoothed = cal_scores

        threshold = float(np.percentile(smoothed, self.cfg.healthy_quantile))

        self._scaler = scaler
        self._model = model
        self._threshold = threshold
        self._healthy_reference = smoothed
        self._rolling_scores = list(smoothed[-50:])
        # Per-channel baseline contribution for attribution.
        self._baseline_contributions = self._build_baseline_contributions(windows)
        self._rolling_contributions = []
        # Start the inference stride counter fresh; the first new window scores
        # once `stride` more samples arrive.
        self._stride_counter = 0

    async def _score_window(
        self, feature_names: list[str], window_length: int
    ) -> tuple[float, float, list[dict[str, float | str]]]:
        """Compute the latest window's raw + smoothed score and per-feature contributors."""

        start = len(self._buffer) - window_length
        window_rows = self._buffer[start:]
        raw = np.asarray([[row[name] for name in feature_names] for row in window_rows], dtype=np.float32)
        scaled = self._scaler.transform(raw).astype(np.float32)
        window = scaled[np.newaxis, ...]
        scoring_cfg = ScoringConfig(
            alpha=self.cfg.scoring_alpha, beta=self.cfg.scoring_beta, batch_size=1
        )
        score = float(
            compute_usad_scores(self._model, window, scoring_cfg, show_progress=False)[0]
        )
        per_feat = self._compute_feature_contributions(scaled)
        contributors = self._rank_contributors(per_feat, feature_names)

        self._rolling_scores.append(score)
        if len(self._rolling_scores) > 500:
            self._rolling_scores = self._rolling_scores[-500:]
        kernel = self.cfg.median_smoothing_window
        recent = self._rolling_scores[-kernel:]
        smoothed = float(np.median(recent)) if recent else score
        return score, smoothed, contributors

    def _alert_level(self, smoothed: float | None, threshold: float | None) -> str:
        """Advance the hysteretic alert state machine by ONE scored window.

        MUTATES streak counters and `self._alert_state`. Call exactly once
        per newly-scored window — never on the unscored in-between sample
        events, or `stride` repeated samples would satisfy the
        "N consecutive windows" trigger from a single window. The unscored
        branch should read `self._alert_state` directly instead.

        - 'alert' fires once `cfg.alert_trigger_streak` consecutive windows
          exceed `threshold`, and clears only after `alert_release_streak`
          consecutive windows fall below `threshold * release_fraction`.
        - 'warn' is the intermediate state: score is above the release band
          but hasn't sustained long enough to declare 'alert'.
        """

        if smoothed is None or threshold is None:
            self._alert_state = "ok"
            self._above_streak = 0
            self._below_streak = 0
            self._warn_streak = 0
            return "ok"

        release_level = threshold * self.cfg.alert_release_fraction
        if smoothed >= threshold:
            self._above_streak += 1
            self._below_streak = 0
            self._warn_streak += 1
        elif smoothed >= release_level:
            self._above_streak = 0
            self._below_streak = 0
            self._warn_streak += 1
        else:
            self._above_streak = 0
            self._below_streak += 1
            self._warn_streak = 0

        if self._alert_state == "alert":
            if self._below_streak >= self.cfg.alert_release_streak:
                self._alert_state = "ok"
        elif self._alert_state == "warn":
            if self._above_streak >= self.cfg.alert_trigger_streak:
                self._alert_state = "alert"
            elif self._below_streak >= self.cfg.alert_release_streak:
                self._alert_state = "ok"
        else:  # ok
            if self._above_streak >= self.cfg.alert_trigger_streak:
                self._alert_state = "alert"
            elif self._warn_streak >= max(2, self.cfg.alert_trigger_streak // 2):
                self._alert_state = "warn"
        return self._alert_state

    def _compute_feature_contributions(self, window_scaled: np.ndarray) -> np.ndarray:
        """Per-feature anomaly contribution for one (T, F) window.

        Returns a (F,) numpy array whose components sum to the same scalar
        as the USAD score computed from this window (within numerical noise).
        """

        x = torch.tensor(window_scaled[np.newaxis, ...], dtype=torch.float32)
        self._model.eval()
        with torch.no_grad():
            recon1, _, _ = self._model(x)
            recon2 = self._model.reconstruct_via_decoder2(recon1)
            mse_ae1 = ((x - recon1) ** 2).mean(dim=(0, 1)).cpu().numpy()
            mse_ae2 = ((x - recon2) ** 2).mean(dim=(0, 1)).cpu().numpy()
        return self.cfg.scoring_alpha * mse_ae1 + self.cfg.scoring_beta * mse_ae2

    def _build_baseline_contributions(self, windows: np.ndarray) -> np.ndarray:
        """Mean per-feature contribution across a batch of calibration windows."""

        x = torch.tensor(windows, dtype=torch.float32)
        self._model.eval()
        per_window: list[np.ndarray] = []
        with torch.no_grad():
            batch = 128
            for i in range(0, x.shape[0], batch):
                chunk = x[i : i + batch]
                recon1, _, _ = self._model(chunk)
                recon2 = self._model.reconstruct_via_decoder2(recon1)
                m1 = ((chunk - recon1) ** 2).mean(dim=1).cpu().numpy()  # (B, F)
                m2 = ((chunk - recon2) ** 2).mean(dim=1).cpu().numpy()
                per_window.append(self.cfg.scoring_alpha * m1 + self.cfg.scoring_beta * m2)
        stacked = np.concatenate(per_window, axis=0)  # (N, F)
        return stacked.mean(axis=0)  # (F,)

    def _rank_contributors(
        self,
        current: np.ndarray,
        feature_names: list[str],
        top_k: int = 5,
    ) -> list[dict[str, float | str]]:
        """Top-K channels by positive deviation from baseline, smoothed."""

        # Maintain a rolling window for stable display.
        self._rolling_contributions.append(current)
        if len(self._rolling_contributions) > 9:
            self._rolling_contributions = self._rolling_contributions[-9:]
        smoothed = np.median(np.stack(self._rolling_contributions, axis=0), axis=0)

        cur_total = max(float(smoothed.sum()), 1e-12)
        cur_pct = smoothed / cur_total * 100.0

        if self._baseline_contributions is None:
            base_pct = np.zeros_like(cur_pct)
        else:
            base_total = max(float(self._baseline_contributions.sum()), 1e-12)
            base_pct = self._baseline_contributions / base_total * 100.0

        delta_pct = cur_pct - base_pct

        descriptions = getattr(self.source.spec, "feature_descriptions", {}) or {}
        actions = getattr(self.source.spec, "suggested_actions", {}) or {}
        order = np.argsort(-delta_pct)
        contributors: list[dict[str, float | str]] = []
        for idx in order[:top_k]:
            name = feature_names[idx]
            contributors.append(
                {
                    "name": name,
                    "label": descriptions.get(name, name),
                    "action": actions.get(name, ""),
                    "delta_pct": float(delta_pct[idx]),
                    "current_pct": float(cur_pct[idx]),
                    "baseline_pct": float(base_pct[idx]),
                }
            )
        return contributors

    def _update_forecast(
        self, event: SensorEvent, smoothed_score: float
    ) -> dict[str, Any] | None:
        """Extrapolate the smoothed-score trend to predict alert crossing time.

        The forecast is label-free: we fit a linear regression to recent
        (asset_seconds, smoothed_score) samples, compute slope + standard
        error, and project to when the line crosses `self._threshold`. The
        method that actually runs on a real edge device — no fleet RUL data
        needed.
        """

        if self._threshold is None:
            return None

        ratio = float(
            getattr(self.source.spec, "simulated_to_asset_seconds", 1.0) or 1.0
        )
        asset_seconds = float(event.elapsed_simulated_seconds) * ratio

        # Drop very-recent duplicates from a paused stream to avoid biasing
        # the slope (timestamps don't advance while pause_event is set).
        if self._forecast_buffer and asset_seconds <= self._forecast_buffer[-1][0]:
            forecast = self._forecast_from_buffer(asset_seconds)
        else:
            self._forecast_buffer.append((asset_seconds, float(smoothed_score)))
            if len(self._forecast_buffer) > self.cfg.forecast_buffer_size:
                self._forecast_buffer = self._forecast_buffer[
                    -self.cfg.forecast_buffer_size :
                ]
            forecast = self._forecast_from_buffer(asset_seconds)
        self._last_forecast = forecast
        return forecast

    def _forecast_from_buffer(self, now_asset_seconds: float) -> dict[str, Any]:
        n = len(self._forecast_buffer)
        if n < self.cfg.forecast_min_samples or self._threshold is None:
            return {
                "status": "warming_up",
                "samples": n,
                "time_to_alert_seconds": None,
            }

        # Center time on `now` so the intercept = current expected score.
        times = np.asarray([t for t, _ in self._forecast_buffer], dtype=np.float64)
        scores = np.asarray([s for _, s in self._forecast_buffer], dtype=np.float64)
        t_centered = times - now_asset_seconds

        mean_t = float(t_centered.mean())
        mean_s = float(scores.mean())
        var_t = float(((t_centered - mean_t) ** 2).sum())
        if var_t < 1e-9:
            return {
                "status": "stable",
                "samples": n,
                "time_to_alert_seconds": None,
                "slope_per_day": 0.0,
            }
        cov = float(((t_centered - mean_t) * (scores - mean_s)).sum())
        slope = cov / var_t
        intercept = mean_s - slope * mean_t  # score at t_centered = 0 (= now)
        residuals = scores - (slope * t_centered + intercept)
        rss = float((residuals ** 2).sum())
        sigma2 = rss / (n - 2) if n > 2 else 0.0
        slope_se = float((sigma2 / var_t) ** 0.5) if var_t > 0 else 0.0
        t_stat = slope / slope_se if slope_se > 1e-12 else 0.0

        current_score = intercept
        slope_per_day = slope * 86400.0

        if current_score >= self._threshold:
            return {
                "status": "above_threshold",
                "samples": n,
                "current_score": current_score,
                "slope_per_day": slope_per_day,
                "time_to_alert_seconds": 0.0,
                "t_stat": t_stat,
            }

        if abs(t_stat) < self.cfg.forecast_significance_t_stat or slope <= 0:
            return {
                "status": "stable",
                "samples": n,
                "current_score": current_score,
                "slope_per_day": slope_per_day,
                "time_to_alert_seconds": None,
                "t_stat": t_stat,
            }

        ttt = (self._threshold - current_score) / slope
        # 95% slope band gives an asymmetric time-to-threshold band.
        slope_hi = slope + 2.0 * slope_se
        slope_lo = max(slope - 2.0 * slope_se, 1e-9)
        ttt_low = (self._threshold - current_score) / slope_hi  # sooner
        ttt_high = (self._threshold - current_score) / slope_lo  # later

        return {
            "status": "trending_up",
            "samples": n,
            "current_score": current_score,
            "slope_per_day": slope_per_day,
            "time_to_alert_seconds": float(ttt),
            "time_to_alert_low_seconds": float(ttt_low),
            "time_to_alert_high_seconds": float(ttt_high),
            "t_stat": t_stat,
        }

    async def _publish_reading(
        self,
        event: SensorEvent,
        score: float | None,
        health: float,
        alert_level: str,
        phase: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "kind": "reading",
            "timestamp": event.timestamp.isoformat() if hasattr(event.timestamp, "isoformat") else str(event.timestamp),
            "index": event.index,
            "elapsed_simulated_seconds": event.elapsed_simulated_seconds,
            "features": event.features,
            "score": score,
            "health": health,
            "alert_level": alert_level,
            "phase": phase,
            "threshold": self._threshold,
        }
        if extra:
            for key, value in extra.items():
                if value is None:
                    continue
                payload[key] = value
        await self.bus.publish("ui.event", payload)

    # ---------- Cycle-based path ----------

    async def _train_async_cycles(
        self,
        cycles: list[np.ndarray],
        window_length: int,
    ) -> None:
        # Pause source so it doesn't burn through its sequence while training
        # runs in the background thread.
        if self._pause_event is not None:
            self._pause_event.set()
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                self._train_blocking_cycles,
                cycles,
                window_length,
            )
        except Exception as exc:
            if self._pause_event is not None:
                self._pause_event.clear()
            await self._broadcast_phase("failed", 0.0, f"training error: {exc}")
            return
        if self._pause_event is not None:
            self._pause_event.clear()
        await self._broadcast_phase(
            "inferring", 1.0, f"threshold = {self._threshold:.3f}"
        )

    def _train_blocking_cycles(
        self,
        cycles: list[np.ndarray],
        window_length: int,
    ) -> None:
        """Cycle-based training: each cycle is one window for Hydraulic; for
        CMAPSS we concatenate cycles and slide a `window_length`-cycle window.
        """

        flat = np.concatenate(cycles, axis=0).astype(np.float32)
        scaler = StandardScaler().fit(flat)
        scaled_flat = scaler.transform(flat).astype(np.float32)

        # Reconstruct per-cycle scaled arrays.
        scaled_cycles: list[np.ndarray] = []
        offset = 0
        for cycle in cycles:
            n = cycle.shape[0]
            scaled_cycles.append(scaled_flat[offset : offset + n])
            offset += n

        # If each cycle is already at least window_length long, treat each
        # cycle as a single training window (Hydraulic). Otherwise concatenate
        # everything and slide a window across (CMAPSS).
        if scaled_cycles[0].shape[0] >= window_length:
            windows = np.stack(
                [c[-window_length:] for c in scaled_cycles], axis=0
            ).astype(np.float32)
        else:
            num_windows = scaled_flat.shape[0] - window_length + 1
            if num_windows < _MIN_CALIBRATION_WINDOWS:
                raise ValueError(
                    f"Not enough calibration cycles for window_length={window_length}"
                    f" (got {scaled_flat.shape[0]} cycle-rows -> {num_windows} "
                    f"windows; need at least {_MIN_CALIBRATION_WINDOWS}). "
                    f"Increase the calibration cycle count."
                )
            windows = np.stack(
                [scaled_flat[i : i + window_length] for i in range(num_windows)],
                axis=0,
            ).astype(np.float32)

        seed_all(self.cfg.seed)
        cfg_model = USADConv1dConfig(
            in_features=windows.shape[2],
            base_channels=self.cfg.base_channels,
            latent_channels=self.cfg.latent_channels,
            downsample_layers=self.cfg.downsample_layers,
        )
        model = USADConv1d(cfg_model)

        train_only, val_only = split_train_validation(windows, val_fraction=0.1)
        train_cfg = TrainingConfig(
            batch_size=min(self.cfg.batch_size, max(8, train_only.shape[0] // 4)),
            epochs=self.cfg.max_epochs,
            learning_rate=self.cfg.learning_rate,
            adv_ramp_epochs=self.cfg.adv_ramp_epochs,
            adv_max_weight=self.cfg.adv_max_weight,
            grad_clip_norm=1.0,
            seed=self.cfg.seed,
        )
        stop_cfg = EarlyStoppingConfig(
            patience=6, min_delta=1e-4, max_epochs=self.cfg.max_epochs, val_fraction=0.1
        )
        train_usad(
            model,
            train_only,
            train_cfg,
            val_windows=val_only,
            early_stopping=stop_cfg,
            show_progress=False,
        )

        scoring_cfg = ScoringConfig(
            alpha=self.cfg.scoring_alpha,
            beta=self.cfg.scoring_beta,
            batch_size=128,
        )
        cal_scores = compute_usad_scores(
            model, windows, scoring_cfg, show_progress=False
        )

        # Smooth with a smaller kernel for cycle-based scoring (fewer points).
        kernel = max(3, min(self.cfg.median_smoothing_window, max(3, windows.shape[0] // 20)))
        if kernel % 2 == 0:
            kernel += 1
        smoothed = (
            apply_median_filter(cal_scores, kernel).astype(np.float32)
            if cal_scores.size
            else cal_scores
        )

        threshold = float(np.percentile(smoothed, self.cfg.healthy_quantile))

        self._scaler = scaler
        self._model = model
        self._threshold = threshold
        self._healthy_reference = smoothed
        self._rolling_scores = list(smoothed[-50:])
        self._baseline_contributions = self._build_baseline_contributions(windows)
        self._rolling_contributions = []
        self._stride_counter = 0

    def _extract_cycle_window(
        self, cycle_buffer: list[np.ndarray], window_length: int
    ) -> np.ndarray | None:
        """Build a (window_length, F) window from the buffer's tail."""

        if not cycle_buffer:
            return None
        last = cycle_buffer[-1]
        if last.shape[0] >= window_length:
            return last[-window_length:]
        # Concatenate the most-recent cycles until total rows >= window_length.
        total = 0
        pieces: list[np.ndarray] = []
        for arr in reversed(cycle_buffer):
            pieces.append(arr)
            total += arr.shape[0]
            if total >= window_length:
                break
        if total < window_length:
            return None
        pieces.reverse()
        concat = np.concatenate(pieces, axis=0)
        return concat[-window_length:]

    async def _score_cycle_window(
        self, window_raw: np.ndarray, feature_names: list[str]
    ) -> tuple[float, float, list[dict[str, float | str]]]:
        scaled = self._scaler.transform(window_raw).astype(np.float32)
        window = scaled[np.newaxis, ...]
        scoring_cfg = ScoringConfig(
            alpha=self.cfg.scoring_alpha, beta=self.cfg.scoring_beta, batch_size=1
        )
        score = float(
            compute_usad_scores(self._model, window, scoring_cfg, show_progress=False)[0]
        )
        per_feat = self._compute_feature_contributions(scaled)
        contributors = self._rank_contributors(per_feat, feature_names)
        self._rolling_scores.append(score)
        if len(self._rolling_scores) > 500:
            self._rolling_scores = self._rolling_scores[-500:]
        kernel = max(3, min(self.cfg.median_smoothing_window, 11))
        if kernel % 2 == 0:
            kernel += 1
        recent = self._rolling_scores[-kernel:]
        smoothed = float(np.median(recent)) if recent else score
        return score, smoothed, contributors

    def _build_diagnosis(
        self,
        alert_level: str,
        forecast: dict[str, Any] | None,
        contributors: list[dict[str, float | str]],
    ) -> dict[str, Any]:
        """Synthesise root cause + urgency + recommended action."""

        urgency, urgency_label = self._urgency_from_state(alert_level, forecast)

        # Healthy / warming up: no detailed diagnosis needed.
        if urgency in ("low", "info"):
            return {
                "urgency": urgency,
                "urgency_label": urgency_label,
                "root_cause": "All systems nominal",
                "recommended_action": "No action required. Continue monitoring.",
                "evidence": [],
            }

        rules = getattr(self.source.spec, "diagnosis_rules", []) or []
        actions = getattr(self.source.spec, "suggested_actions", {}) or {}
        descriptions = getattr(self.source.spec, "feature_descriptions", {}) or {}

        positive = [
            c for c in contributors
            if isinstance(c.get("delta_pct"), (int, float)) and c["delta_pct"] > 0
        ]
        top_names = {c["name"] for c in positive[:5]}

        matched: dict | None = None
        for rule in rules:
            required = rule.get("requires", []) or []
            if required and all(name in top_names for name in required):
                matched = rule
                break

        evidence: list[str] = []
        for c in positive[:3]:
            label = c.get("label") or c.get("name")
            delta = c.get("delta_pct", 0.0)
            evidence.append(f"{label} +{float(delta):.1f} pts from baseline")

        if matched is not None:
            return {
                "urgency": urgency,
                "urgency_label": urgency_label,
                "root_cause": matched.get("name", "Multi-channel anomaly"),
                "recommended_action": matched.get("action", ""),
                "evidence": evidence,
                "matched_rule": matched.get("name", ""),
            }

        # Fallback: single top contributor + its existing suggested action.
        if positive:
            top = positive[0]
            top_name = top.get("name", "")
            label = top.get("label") or top_name or "Unknown channel"
            action = actions.get(top_name) or top.get("action") or "Investigate the flagged channel."
            return {
                "urgency": urgency,
                "urgency_label": urgency_label,
                "root_cause": f"Deviation localised to {label}",
                "recommended_action": action,
                "evidence": evidence,
                "matched_rule": "",
            }

        return {
            "urgency": urgency,
            "urgency_label": urgency_label,
            "root_cause": "Anomaly detected without clear localisation",
            "recommended_action": "Run full asset inspection at next maintenance window.",
            "evidence": [],
            "matched_rule": "",
        }

    def _urgency_from_state(
        self,
        alert_level: str,
        forecast: dict[str, Any] | None,
    ) -> tuple[str, str]:
        status = (forecast or {}).get("status")
        ttt = (forecast or {}).get("time_to_alert_seconds")

        if alert_level == "alert" and status == "above_threshold":
            return "critical", "Critical — investigate now"
        if alert_level == "alert":
            return "high", "High — schedule within 24 hours"
        if status == "trending_up" and ttt is not None and ttt < 7 * 86400:
            return "high", "High — alert projected within 1 week"
        if alert_level == "warn":
            return "medium", "Medium — schedule routine check"
        if status == "trending_up" and ttt is not None and ttt < 30 * 86400:
            return "medium", "Medium — alert projected within 1 month"
        if status == "warming_up":
            return "info", "Info — system calibrating"
        return "low", "Low — no action required"

    async def _broadcast_phase(self, name: str, progress: float, detail: str) -> None:
        self._phase = DevicePhase(name=name, progress=progress, detail=detail)
        await self.bus.publish(
            "ui.event",
            {
                "kind": "phase",
                "phase": name,
                "progress": progress,
                "detail": detail,
            },
        )
