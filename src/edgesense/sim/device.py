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
from ..models import RULHead, USADConv1d, USADConv1dConfig
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
        self._cycle_buffer: list[np.ndarray] = []
        self._cycle_rul_targets: list[float] = []
        self._scaler: StandardScaler | None = None
        self._model: USADConv1d | None = None
        self._rul_head: RULHead | None = None
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
                self._cycle_rul_targets.clear()
                self._rolling_scores = []
                self._window_count = 0
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
                self._training_task = asyncio.create_task(
                    self._train_async(feature_names, window_length, stride)
                )

        elif phase_name == "training":
            await self._publish_reading(event, score=None, health=100.0, alert_level="ok", phase="training")

        elif phase_name == "inferring":
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
        self._cycle_buffer.append(event.cycle_features)
        true_rul = event.metadata.get("true_rul")
        self._cycle_rul_targets.append(
            float(true_rul) if true_rul is not None else float("nan")
        )
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
                rul_snapshot = list(self._cycle_rul_targets)
                self._training_task = asyncio.create_task(
                    self._train_async_cycles(cycles_snapshot, window_length, rul_snapshot)
                )

        elif phase_name == "training":
            await self._publish_reading(event, score=None, health=100.0, alert_level="ok", phase="training")

        elif phase_name == "inferring":
            window = self._extract_cycle_window(self._cycle_buffer, window_length)
            if window is None or self._scaler is None or self._model is None:
                await self._publish_reading(event, score=None, health=100.0, alert_level="ok", phase="inferring")
                return
            score, smoothed = await self._score_cycle_window(window)
            health = float(
                health_score(
                    np.asarray([smoothed], dtype=np.float32),
                    self._healthy_reference,
                    self._threshold,
                )[0]
            )
            alert_level = self._alert_level(smoothed, self._threshold)
            rul_pred = self._predict_rul(window) if self._rul_head is not None else None
            await self._publish_reading(
                event,
                score=smoothed,
                health=health,
                alert_level=alert_level,
                phase="inferring",
                extra={
                    "true_anomaly": event.metadata.get("is_anomaly"),
                    "true_rul": event.metadata.get("true_rul"),
                    "unit_id": event.metadata.get("unit_id"),
                    "unit_cycle": event.metadata.get("unit_cycle"),
                    "rul_pred": rul_pred,
                },
            )

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
        rul_targets: list[float] | None = None,
    ) -> None:
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                self._train_blocking_cycles,
                cycles,
                window_length,
                rul_targets or [],
            )
        except Exception as exc:
            await self._broadcast_phase("failed", 0.0, f"training error: {exc}")
            return
        detail = f"threshold = {self._threshold:.3f}"
        if self._rul_head is not None:
            detail += " | RUL head ready"
        await self._broadcast_phase("inferring", 1.0, detail)

    def _train_blocking_cycles(
        self,
        cycles: list[np.ndarray],
        window_length: int,
        rul_targets: list[float],
    ) -> None:
        """Cycle-based training: each cycle is one window for Hydraulic; for
        CMAPSS we concatenate cycles and slide a `window_length`-cycle window.

        If `rul_targets` are provided and the source declares an "anomaly+rul"
        output kind, a RULHead is also trained on top of the frozen USAD
        encoder using the per-window RUL targets.
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
            # Per-cycle layout: window i corresponds to cycle i.
            window_rul = (
                np.asarray(rul_targets, dtype=np.float32) if rul_targets else None
            )
        else:
            num_windows = scaled_flat.shape[0] - window_length + 1
            if num_windows < 32:
                raise ValueError(
                    f"Not enough calibration cycles for window_length={window_length}"
                    f" (got {scaled_flat.shape[0]} cycle-rows total)."
                )
            windows = np.stack(
                [scaled_flat[i : i + window_length] for i in range(num_windows)],
                axis=0,
            ).astype(np.float32)
            # Sliding layout: for single-step cycles (CMAPSS), row i = cycle i,
            # so window i ends at cycle (window_length - 1 + i).
            if rul_targets and len(rul_targets) >= window_length:
                window_rul = np.asarray(
                    [rul_targets[window_length - 1 + i] for i in range(num_windows)],
                    dtype=np.float32,
                )
            else:
                window_rul = None

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
        if kernel >= 3:
            half = kernel // 2
            padded = np.pad(cal_scores, (half, half), mode="edge")
            smoothed = np.array(
                [np.median(padded[i : i + kernel]) for i in range(len(cal_scores))],
                dtype=np.float32,
            )
        else:
            smoothed = cal_scores

        threshold = float(np.percentile(smoothed, self.cfg.healthy_quantile))

        self._scaler = scaler
        self._model = model
        self._threshold = threshold
        self._healthy_reference = smoothed
        self._rolling_scores = list(smoothed[-50:])
        self._window_count = 0

        # Optional RUL head, trained on top of the frozen encoder.
        if (
            getattr(self.source.spec, "output_kind", "anomaly") == "anomaly+rul"
            and window_rul is not None
            and np.isfinite(window_rul).all()
        ):
            self._rul_head = self._train_rul_head(model, windows, window_rul)

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

    async def _score_cycle_window(self, window_raw: np.ndarray) -> tuple[float, float]:
        scaled = self._scaler.transform(window_raw).astype(np.float32)
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
        kernel = max(3, min(self.cfg.median_smoothing_window, 11))
        if kernel % 2 == 0:
            kernel += 1
        recent = self._rolling_scores[-kernel:]
        smoothed = float(np.median(recent)) if recent else score
        return score, smoothed

    def _train_rul_head(
        self,
        model: USADConv1d,
        windows: np.ndarray,
        window_rul: np.ndarray,
    ) -> RULHead:
        """Train a small RUL regression head on top of the frozen USAD encoder."""

        from torch import nn

        device = torch.device("cpu")
        x = torch.tensor(windows, dtype=torch.float32, device=device)
        y = torch.tensor(window_rul, dtype=torch.float32, device=device)

        # Pre-compute latents once (encoder is frozen).
        model.eval()
        with torch.no_grad():
            latents: list[torch.Tensor] = []
            batch = 256
            for i in range(0, x.shape[0], batch):
                latents.append(model.encode(x[i : i + batch]))
            latents_all = torch.cat(latents, dim=0)

        head = RULHead(latent_channels=model.config.latent_channels, hidden_dim=64, dropout=0.2)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.MSELoss()

        rng = np.random.default_rng(self.cfg.seed)
        n = latents_all.shape[0]
        idx = np.arange(n)
        rng.shuffle(idx)
        train_idx = idx[: int(n * 0.9)]
        val_idx = idx[int(n * 0.9) :]
        x_train = latents_all[train_idx]
        y_train = y[train_idx]
        x_val = latents_all[val_idx]
        y_val = y[val_idx]

        best_state = None
        best_val = float("inf")
        patience_counter = 0
        for epoch in range(40):
            head.train()
            order = np.arange(x_train.shape[0])
            rng.shuffle(order)
            for start in range(0, len(order), 128):
                batch_idx = order[start : start + 128]
                preds = head(x_train[batch_idx])
                loss = criterion(preds, y_train[batch_idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step()
            head.eval()
            with torch.no_grad():
                val_loss = float(((head(x_val) - y_val) ** 2).mean())
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 6:
                    break
        if best_state is not None:
            head.load_state_dict(best_state)
        head.eval()
        return head

    def _predict_rul(self, window_raw: np.ndarray) -> float | None:
        if self._rul_head is None or self._model is None or self._scaler is None:
            return None
        scaled = self._scaler.transform(window_raw).astype(np.float32)
        window = torch.tensor(scaled[np.newaxis, ...], dtype=torch.float32)
        with torch.no_grad():
            latent = self._model.encode(window)
            pred = float(self._rul_head(latent).item())
        return max(0.0, pred)

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
