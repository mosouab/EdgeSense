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

from .bus import EventBus
from .device import DeviceConfig, EdgeDevice
from .source import DataSource, get_source, list_available_sources

LOG = logging.getLogger("edgesense.sim")

STATIC_DIR = Path(__file__).resolve().parent / "static"


class StartRequest(BaseModel):
    source: str = "metropt"
    speed: float = 60.0
    calibration_samples: int = 30_000


class SpeedRequest(BaseModel):
    speed: float


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

    async def start(self, source_name: str, speed: float, calibration_samples: int) -> None:
        await self.stop()
        self.stop_event = asyncio.Event()
        self.pause_event = asyncio.Event()
        self.speed = speed
        self.current_source = source_name

        source = get_source(source_name)
        device_cfg = DeviceConfig(calibration_samples=calibration_samples)
        self.device = EdgeDevice(self.bus, source, device_cfg)
        self.task = asyncio.create_task(self._run(source))

    async def stop(self) -> None:
        if self.task is not None and not self.task.done():
            self.stop_event.set()
            try:
                await asyncio.wait_for(self.task, timeout=2.0)
            except asyncio.TimeoutError:
                self.task.cancel()
            except Exception:
                pass
        self.task = None
        self.device = None
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
        # Pass a getter so /speed updates take effect mid-stream instead of
        # only at next start.
        def get_speed() -> float:
            return self.speed

        async def adaptive_stream():
            async for ev in source.stream(get_speed, self.stop_event, self.pause_event):
                yield ev

        try:
            await self.device.run(adaptive_stream())
        except asyncio.CancelledError:
            pass
        except Exception:
            LOG.exception("Simulation crashed.")


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
