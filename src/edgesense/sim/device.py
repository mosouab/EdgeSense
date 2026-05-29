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

from ..health import health_score
from ..models import USADConv1d, USADConv1dConfig
from ..scoring import ScoringConfig, compute_usad_scores
from ..training import EarlyStoppingConfig, TrainingConfig, seed_all, split_train_validation, train_usad
from .bus import EventBus
from .source import DataSource, SensorEvent


@dataclass
class DeviceConfig:
    calibration_samples: int = 30_000
    healthy_quantile: float = 99.0
    scoring_alpha: float = 0.3
    scoring_beta: float = 0.7
    median_smoothing_window: int = 11
    base_channels: int = 32
    latent_channels: int = 64
    downsample_layers: int = 2
    batch_size: int = 256
    learning_rate: float = 1e-3
    max_epochs: int = 25
    adv_ramp_epochs: int = 15
    adv_max_weight: float = 0.3
    seed: int = 42


@dataclass
class DevicePhase:
    """Snapshot of the device's current lifecycle state, broadcast to UI."""

    name: str
    progress: float
    detail: str = ""


class EdgeDevice:
    """Run the calibration -> train -> infer lifecycle for one DataSource."""

    def __init__(self, bus: EventBus, source: DataSource, cfg: DeviceConfig) -> None:
        self.bus = bus
        self.source = source
        self.cfg = cfg
        self._buffer: list[dict[str, float]] = []
        self._scaler: StandardScaler | None = None
        self._model: USADConv1d | None = None
        self._threshold: float | None = None
        self._healthy_reference: np.ndarray | None = None
        self._rolling_scores: list[float] = []
        self._phase = DevicePhase(name="awaiting", progress=0.0)
        self._window_count = 0
        self._training_task: asyncio.Task | None = None

    @property
    def phase(self) -> DevicePhase:
        return self._phase

    async def run(self, source_stream) -> None:
        """Consume the source stream end-to-end."""

        spec = self.source.spec
        feature_names = spec.feature_names
        window_length = spec.window_length
        stride = spec.stride
        calibration_target = self.cfg.calibration_samples
        await self._broadcast_phase("calibrating", 0.0, f"collecting {calibration_target:,} samples")

        async for event in source_stream:
            if event.metadata.get("jumped"):
                # Source teleported. Drop any history that pre-dates the jump
                # so the next scored window contains only post-jump data.
                self._buffer.clear()
                self._rolling_scores = []
                self._window_count = 0
                await self._broadcast_phase(
                    self._phase.name,
                    self._phase.progress,
                    f"jumped to row {event.index} ({event.timestamp})",
                )
            self._buffer.append(event.features)
            phase_name = self._phase.name

            if phase_name == "calibrating":
                progress = min(len(self._buffer) / max(calibration_target, 1), 1.0)
                if len(self._buffer) % 200 == 0:
                    await self._broadcast_phase("calibrating", progress, f"{len(self._buffer):,} / {calibration_target:,} samples")
                await self._publish_reading(event, score=None, health=100.0, alert_level="ok", phase="calibrating")
                if len(self._buffer) >= calibration_target and self._training_task is None:
                    await self._broadcast_phase("training", 0.0, "fitting scaler + USAD model")
                    self._training_task = asyncio.create_task(self._train_async(feature_names, window_length, stride))

            elif phase_name == "training":
                # While training runs in a background thread, keep echoing readings unchanged.
                await self._publish_reading(event, score=None, health=100.0, alert_level="ok", phase="training")

            elif phase_name == "inferring":
                # Score windows on a rolling basis at stride cadence.
                if len(self._buffer) - self._window_count * stride >= window_length:
                    score, smoothed = await self._score_window(feature_names, window_length)
                    self._window_count += 1
                    health = float(
                        health_score(
                            np.asarray([smoothed], dtype=np.float32),
                            self._healthy_reference,
                            self._threshold,
                        )[0]
                    )
                    alert_level = self._alert_level(smoothed, self._threshold)
                    await self._publish_reading(event, score=smoothed, health=health, alert_level=alert_level, phase="inferring")
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
                    alert_level = self._alert_level(last_smoothed, self._threshold)
                    await self._publish_reading(event, score=last_smoothed, health=last_health, alert_level=alert_level, phase="inferring")

        if self._training_task is not None:
            await self._training_task
        await self._broadcast_phase("finished", 1.0, "stream complete")

    async def _train_async(self, feature_names: list[str], window_length: int, stride: int) -> None:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, self._train_blocking, list(feature_names), window_length, stride
            )
        except Exception as exc:
            await self._broadcast_phase("failed", 0.0, f"training error: {exc}")
            return
        await self._broadcast_phase(
            "inferring", 1.0, f"threshold = {self._threshold:.3f}"
        )

    def _train_blocking(self, feature_names: list[str], window_length: int, stride: int) -> None:
        """Heavy lifting: fit scaler, build windows, train USAD, set threshold."""

        df = np.asarray([[row[name] for name in feature_names] for row in self._buffer], dtype=np.float32)
        # Interpolate-then-fillna is unnecessary here because the source never emits NaN.
        scaler = StandardScaler().fit(df)
        scaled = scaler.transform(df).astype(np.float32)

        # Sliding windows over the calibration buffer.
        num_windows = max(0, (len(scaled) - window_length) // stride + 1)
        if num_windows < 32:
            # Not enough data; bail to a tiny fallback (rare in practice).
            num_windows = max(num_windows, 1)
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
        if kernel >= 3 and kernel % 2 == 1:
            half = kernel // 2
            padded = np.pad(cal_scores, (half, half), mode="edge")
            smoothed = np.array([
                np.median(padded[i : i + kernel]) for i in range(len(cal_scores))
            ], dtype=np.float32)
        else:
            smoothed = cal_scores

        threshold = float(np.percentile(smoothed, self.cfg.healthy_quantile))

        self._scaler = scaler
        self._model = model
        self._threshold = threshold
        self._healthy_reference = smoothed
        self._rolling_scores = list(smoothed[-50:])
        # Reset window counter so inference scoring picks up from current buffer head.
        # We've already consumed the calibration windows; new windows start AFTER buffer head.
        self._window_count = (len(self._buffer) - window_length) // stride + 1

    async def _score_window(self, feature_names: list[str], window_length: int) -> tuple[float, float]:
        """Compute the latest window's raw + smoothed score using the trained model."""

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

        self._rolling_scores.append(score)
        if len(self._rolling_scores) > 500:
            self._rolling_scores = self._rolling_scores[-500:]
        kernel = self.cfg.median_smoothing_window
        recent = self._rolling_scores[-kernel:]
        smoothed = float(np.median(recent)) if recent else score
        return score, smoothed

    def _alert_level(self, smoothed: float | None, threshold: float | None) -> str:
        if smoothed is None or threshold is None:
            return "ok"
        if smoothed >= threshold:
            return "alert"
        if smoothed >= threshold * 0.6:
            return "warn"
        return "ok"

    async def _publish_reading(
        self,
        event: SensorEvent,
        score: float | None,
        health: float,
        alert_level: str,
        phase: str,
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
        await self.bus.publish("ui.event", payload)

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
