"""FastAPI app that wires sensor + device + WebSocket fan-out."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..cmms import MockCmmsClient, build_work_request
from ..feedback import FeedbackStore, build_feedback_record
from .bus import EventBus
from .device import DeviceConfig, EdgeDevice
from .source import DataSource, get_source, list_available_sources

WORK_ORDER_DIR = Path("reports/work_orders")
FEEDBACK_DIR = Path("reports/feedback")

LOG = logging.getLogger("edgesense.sim")

STATIC_DIR = Path(__file__).resolve().parent / "static"


class StartRequest(BaseModel):
    source: str = "metropt"
    speed: float = 60.0
    calibration_samples: int = 30_000


class SpeedRequest(BaseModel):
    speed: float


class JumpRequest(BaseModel):
    failure_id: int | None = None
    index: int | None = None


class FeedbackRequest(BaseModel):
    episode_id: str
    verdict: str               # "false_positive" | "confirmed"
    note: str | None = None


class WorkRequestRequest(BaseModel):
    diagnosis: dict
    contributors: list[dict] | None = None
    forecast: dict | None = None
    asset_id: str
    asset_label: str | None = None
    score: float | None = None
    threshold: float | None = None
    elapsed_simulated_seconds: float | None = None
    requested_by: str | None = None
    preview_only: bool = False


class SimulationState:
    """Owns the running sim task plus its control flags. Singleton-per-process."""

    def __init__(self) -> None:
        self.bus = EventBus()
        self.task: asyncio.Task | None = None
        self.stop_event: asyncio.Event = asyncio.Event()
        self.pause_event: asyncio.Event = asyncio.Event()
        self.speed: float = 60.0
        self.current_source: str | None = None
        self.device: EdgeDevice | None = None
        self.source: DataSource | None = None
        self._seek_to: int | None = None

    async def start(self, source_name: str, speed: float, calibration_samples: int) -> None:
        await self.stop()
        self.stop_event = asyncio.Event()
        self.pause_event = asyncio.Event()
        self.speed = speed
        self.current_source = source_name

        source = get_source(source_name, calibration_size=calibration_samples)
        device_cfg = DeviceConfig(calibration_samples=calibration_samples)
        # Pass the same pause_event so the device can stop the source while
        # it trains (otherwise the source can exhaust before inference).
        self.device = EdgeDevice(self.bus, source, device_cfg, pause_event=self.pause_event)
        self.source = source
        self._seek_to = None
        self.task = asyncio.create_task(self._run(source))

    async def stop(self) -> None:
        device = self.device
        if self.task is not None and not self.task.done():
            self.stop_event.set()
            try:
                await asyncio.wait_for(self.task, timeout=2.0)
            except asyncio.TimeoutError:
                self.task.cancel()
            except Exception:
                pass
        # Cancelling the run task does not reach the training executor thread.
        # Wait for it explicitly so a subsequent start() can't run two torch
        # trainings concurrently (a real stutter risk on a laptop CPU).
        if device is not None:
            await device.await_training()
        self.task = None
        self.device = None
        self.source = None
        self._seek_to = None
        self.current_source = None

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.1, min(speed, 5000.0))

    def toggle_pause(self) -> bool:
        if self.pause_event.is_set():
            self.pause_event.clear()
            return False
        self.pause_event.set()
        return True

    async def _run(self, source: DataSource) -> None:
        # Pass getters so /speed and /jump take effect mid-stream.
        def get_speed() -> float:
            return self.speed

        def consume_seek() -> int | None:
            seek = self._seek_to
            self._seek_to = None
            return seek

        async def adaptive_stream():
            async for ev in source.stream(
                get_speed, consume_seek, self.stop_event, self.pause_event
            ):
                yield ev

        try:
            await self.device.run(adaptive_stream())
        except asyncio.CancelledError:
            pass
        except Exception:
            LOG.exception("Simulation crashed.")

    def request_jump(self, index: int) -> None:
        self._seek_to = max(0, int(index))


state = SimulationState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await state.stop()


app = FastAPI(title="EdgeSense Simulation", lifespan=lifespan)


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/sources")
async def get_sources() -> dict[str, Any]:
    return {"sources": list_available_sources()}


@app.get("/failures")
async def get_failures(source: str | None = None) -> dict[str, Any]:
    """Failure markers for `?source=NAME` (default: the running source, else metropt)."""

    if source is not None:
        try:
            transient = get_source(source)
        except ValueError:
            return {"failures": [], "error": f"unknown source: {source}"}
        try:
            markers = transient.failure_markers()
        except Exception:
            LOG.exception("Failed to compute failure markers for %s", source)
            return {"failures": []}
        return {"source": source, "failures": [m.__dict__ for m in markers]}

    active = state.source
    if active is None:
        try:
            active = get_source("metropt")
        except Exception:
            return {"failures": []}
    try:
        markers = active.failure_markers()
    except Exception:
        LOG.exception("Failed to compute failure markers")
        return {"failures": []}
    return {
        "source": active.spec.name,
        "failures": [m.__dict__ for m in markers],
    }


@app.post("/jump")
async def jump(req: JumpRequest) -> dict[str, Any]:
    if state.source is None or state.device is None:
        return {"status": "error", "detail": "simulation not running"}
    if state.device.phase.name != "inferring":
        return {
            "status": "rejected",
            "detail": f"jumps only allowed while inferring (phase = {state.device.phase.name})",
        }

    index: int | None = req.index
    if index is None and req.failure_id is not None:
        markers = state.source.failure_markers()
        for m in markers:
            if m.id == req.failure_id:
                index = m.jump_index
                break
    if index is None:
        return {"status": "error", "detail": "failure_id or index required"}

    state.request_jump(index)
    return {"status": "ok", "index": index}


@app.post("/start")
async def start(req: StartRequest) -> dict[str, Any]:
    await state.start(req.source, req.speed, req.calibration_samples)
    return {"status": "started", "source": req.source, "speed": req.speed}


@app.post("/stop")
async def stop() -> dict[str, Any]:
    await state.stop()
    return {"status": "stopped"}


@app.post("/pause")
async def pause() -> dict[str, Any]:
    paused = state.toggle_pause()
    return {"status": "paused" if paused else "running"}


@app.post("/speed")
async def speed(req: SpeedRequest) -> dict[str, Any]:
    state.set_speed(req.speed)
    return {"status": "ok", "speed": state.speed}


@app.post("/work_request")
async def post_work_request(req: WorkRequestRequest) -> dict[str, Any]:
    """Build a CMMS-ready work request from the supplied diagnosis context.

    With `preview_only=True`, returns the constructed payload without
    submitting (used by the dashboard's preview modal). Otherwise routes
    through the configured CMMS adapter (currently MockCmmsClient, which
    writes JSON under reports/work_orders/).
    """

    try:
        work_request = build_work_request(
            diagnosis=req.diagnosis or {},
            contributors=req.contributors or [],
            forecast=req.forecast,
            asset_id=req.asset_id,
            asset_label=req.asset_label or req.asset_id,
            score=req.score,
            threshold=req.threshold,
            elapsed_simulated_seconds=req.elapsed_simulated_seconds,
            requested_by=req.requested_by or "EdgeSense Edge Device",
        )
    except Exception as exc:
        LOG.exception("Failed to build work request")
        return {"status": "error", "detail": str(exc)}

    payload = work_request.to_dict()
    if req.preview_only:
        return {"status": "preview", "request": payload}

    try:
        client = MockCmmsClient(WORK_ORDER_DIR)
        result = client.submit(work_request)
    except Exception as exc:
        LOG.exception("CMMS submission failed")
        return {"status": "error", "detail": str(exc), "request": payload}

    return {"status": "submitted", **result.to_dict()}


@app.post("/feedback")
async def post_feedback(req: FeedbackRequest) -> dict[str, Any]:
    """Record an operator verdict on an alert episode.

    Snapshots the episode authoritatively from the running device, persists an
    append-only JSONL record, dismisses the alert on a false positive
    (force_release), and broadcasts a 'feedback' event so the UI can react.
    """

    source = state.current_source or "unknown"
    episode = state.device.get_episode(req.episode_id) if state.device is not None else None

    try:
        record = build_feedback_record(
            episode_id=req.episode_id,
            source=source,
            verdict=req.verdict,
            note=req.note or "",
            episode=episode,
        )
    except ValueError as exc:
        return {"status": "error", "detail": str(exc)}

    try:
        FeedbackStore(FEEDBACK_DIR).append(record)
    except Exception as exc:
        LOG.exception("Failed to persist feedback")
        return {"status": "error", "detail": str(exc)}

    released_id = None
    collected = None
    if req.verdict == "false_positive" and state.device is not None:
        released_id = state.device.force_release()
        # Layer 2: collect this episode's windows as operator-confirmed healthy.
        final_episode = state.device.get_episode(req.episode_id)
        collected = state.device.collect_dismissed_windows(final_episode)

    await state.bus.publish(
        "ui.event",
        {
            "kind": "feedback",
            "feedback_id": record.feedback_id,
            "episode_id": req.episode_id,
            "verdict": req.verdict,
            "released_episode_id": released_id,
            "collected": collected,
            "adaptation": state.device.adaptation_state() if state.device is not None else None,
        },
    )
    return {"status": "recorded", "collected": collected, **record.to_dict()}


@app.post("/recalibrate")
async def post_recalibrate() -> dict[str, Any]:
    """Retrain the model on calibration + operator-dismissed windows (Layer 2)."""

    if state.device is None:
        return {"status": "error", "detail": "no running simulation"}
    result = await state.device.recalibrate()
    if result.get("status") == "recalibrated":
        await state.bus.publish(
            "ui.event",
            {"kind": "recalibrated", **result, "adaptation": state.device.adaptation_state()},
        )
    return result


@app.post("/revert")
async def post_revert() -> dict[str, Any]:
    """Restore the previous model + threshold snapshot (Layer 2)."""

    if state.device is None:
        return {"status": "error", "detail": "no running simulation"}
    result = state.device.revert()
    if result.get("status") == "reverted":
        await state.bus.publish(
            "ui.event",
            {"kind": "reverted", **result, "adaptation": state.device.adaptation_state()},
        )
    return result


@app.get("/adaptation")
async def get_adaptation() -> dict[str, Any]:
    if state.device is None:
        return {"adaptation": None}
    return {"adaptation": state.device.adaptation_state()}


@app.get("/feedback")
async def get_feedback(source: str | None = None) -> dict[str, Any]:
    return {"feedback": FeedbackStore(FEEDBACK_DIR).list(source)}


@app.get("/status")
async def status() -> dict[str, Any]:
    return {
        "running": state.task is not None and not state.task.done(),
        "source": state.current_source,
        "speed": state.speed,
        "paused": state.pause_event.is_set(),
        "phase": state.device.phase.__dict__ if state.device is not None else None,
    }


@app.websocket("/ws")
async def websocket(ws: WebSocket) -> None:
    await ws.accept()
    queue = await state.bus.subscribe("ui.event", maxsize=2048)
    try:
        while True:
            event = await queue.get()
            await ws.send_text(json.dumps(event, default=str))
    except WebSocketDisconnect:
        pass
    finally:
        await state.bus.unsubscribe("ui.event", queue)
