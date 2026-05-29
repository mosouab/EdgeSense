"""Minimal in-process async pub/sub event bus.

Designed so that the sensor publisher, the device publisher/subscriber, and
the WebSocket fan-out subscriber don't know anything about each other. The
interface is intentionally close enough to MQTT that swapping the
implementation later is contained.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any


class EventBus:
    """One-to-many async topic bus. Each subscriber has its own queue."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def publish(self, topic: str, event: Any) -> None:
        """Fan an event out to every subscriber on `topic`.

        Slow consumers are tolerated: if a subscriber's queue is full we
        drop the oldest message so the latest state is always available.
        """

        async with self._lock:
            subs = list(self._subscribers.get(topic, ()))
        for queue in subs:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await queue.put(event)

    async def subscribe(self, topic: str, maxsize: int = 1024) -> asyncio.Queue:
        """Open a queue and return it."""

        queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        async with self._lock:
            self._subscribers[topic].add(queue)
        return queue

    async def unsubscribe(self, topic: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.get(topic, set()).discard(queue)
