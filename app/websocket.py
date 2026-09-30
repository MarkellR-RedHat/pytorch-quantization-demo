"""WebSocket connection manager for real-time updates to the presenter screen"""

import asyncio
import json
import logging

from fastapi import WebSocket

logger = logging.getLogger(__name__)

SEND_TIMEOUT_S = 1.0


class ConnectionManager:
    """Tracks presenter sockets; one slow client never stalls a broadcast"""

    def __init__(self):
        self.presenters: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.presenters.add(websocket)
        logger.info(f"New presenter connection. Presenters: {len(self.presenters)}")

    def disconnect(self, websocket: WebSocket):
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

    async def broadcast(self, message: dict):
        connections = list(self.presenters)
        if not connections:
            return
        text = json.dumps(message, default=str)
        results = await asyncio.gather(*(self._send_text(ws, text) for ws in connections))
        for ws in results:
            if ws is not None:
                self.disconnect(ws)

    def get_presenter_count(self) -> int:
        return len(self.presenters)
