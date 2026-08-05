"""WebSocket surface for live cycle updates (SPEC §9).

Read-only, like the rest of the API: the socket pushes state outward and
accepts nothing that could change a decision. A client that disconnects loses
nothing, because every message is also available from a REST endpoint — the
socket is a latency optimization, not a source of truth.

The broadcaster is deliberately independent of the trading loop. The loop
publishes and moves on; a slow or absent subscriber can never block a
rebalance.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.api.schemas import DISCLAIMER

#: Bounded so a subscriber that stops reading cannot grow memory without limit.
QUEUE_LIMIT = 256


class Socket(Protocol):
    """The subset of a WebSocket this module uses."""

    async def accept(self) -> None: ...
    async def send_text(self, data: str) -> None: ...


@dataclass
class Broadcaster:
    """Fan-out of cycle events to connected dashboards."""

    subscribers: list[asyncio.Queue[str]] = field(default_factory=list)

    def subscribe(self) -> asyncio.Queue[str]:
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=QUEUE_LIMIT)
        self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[str]) -> None:
        if queue in self.subscribers:
            self.subscribers.remove(queue)

    def publish(self, event: str, payload: dict[str, Any]) -> int:
        """Push an event to every subscriber. Never blocks, never raises.

        A full queue means that client stopped reading; the message is dropped
        for them alone. Blocking the publisher — the trading loop — to wait on a
        dashboard would be the wrong trade every time.
        """
        message = json.dumps(
            {"event": event, "payload": payload, "disclaimer": DISCLAIMER}, sort_keys=True
        )
        delivered = 0
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(message)
                delivered += 1
            except asyncio.QueueFull:
                continue
        return delivered

    @property
    def subscriber_count(self) -> int:
        return len(self.subscribers)


async def stream(socket: Socket, broadcaster: Broadcaster) -> None:
    """Accept a socket and forward published events until it is closed."""
    await socket.accept()
    queue = broadcaster.subscribe()
    try:
        while True:
            await socket.send_text(await queue.get())
    finally:
        broadcaster.unsubscribe(queue)
