"""WebSocket connection manager for real-time updates"""

import asyncio
import json
import logging
from typing import Literal

from fastapi import WebSocket

logger = logging.getLogger(__name__)

SEND_TIMEOUT_S = 1.0
Target = Literal["all", "presenter", "audience"]


class ConnectionManager:
    """Tracks audience and presenter sockets; one slow client never stalls a broadcast"""

    def __init__(self, max_connections: int = 2000):
        self.max_connections = max_connections
        self.audience: set[WebSocket] = set()
        self.presenters: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket, is_presenter: bool = False) -> bool:
        if not is_presenter and len(self.audience) >= self.max_connections:
            await websocket.close(code=1013)
            return False
        await websocket.accept()
        (self.presenters if is_presenter else self.audience).add(websocket)
        logger.info(f"New connection. Audience: {len(self.audience)}, presenters: {len(self.presenters)}")
        return True

    def disconnect(self, websocket: WebSocket):
        self.audience.discard(websocket)
        self.presenters.discard(websocket)

    async def _send_text(self, websocket: WebSocket, text: str) -> WebSocket | None:
        try:
            await asyncio.wait_for(websocket.send_text(text), SEND_TIMEOUT_S)
            return None
        except Exception as e:  # a dead or slow socket is dropped, never retried
            logger.debug(f"Dropping client after send failure: {e!r}")
            return websocket

    async def send(self, websocket: WebSocket, message: dict):
        failed = await self._send_text(websocket, json.dumps(message, default=str))
        if failed:
            self.disconnect(failed)

    async def broadcast(self, message: dict, target: Target = "all"):
        if target == "presenter":
            connections = list(self.presenters)
        elif target == "audience":
            connections = list(self.audience)
        else:
            connections = list(self.presenters | self.audience)
        if not connections:
            return
        text = json.dumps(message, default=str)
        results = await asyncio.gather(*(self._send_text(ws, text) for ws in connections))
        for ws in results:
            if ws is not None:
                self.disconnect(ws)

    async def broadcast_metrics(self, metrics: dict):
        await self.broadcast({"type": "metrics_update", "data": metrics})

    async def broadcast_state(self, state: dict):
        await self.broadcast({"type": "state_update", "data": state})

    async def notify_presenter(self, event_type: str, data: dict):
        await self.broadcast({"type": event_type, "data": data}, target="presenter")

    def get_connection_count(self) -> int:
        """Audience connections only; the presenter screen is not a participant."""
        return len(self.audience)

    def get_presenter_count(self) -> int:
        return len(self.presenters)
