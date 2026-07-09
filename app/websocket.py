"""WebSocket connection manager for real-time updates"""

import json
import asyncio
import logging
from typing import List, Set
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.presenter_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket, is_presenter: bool = False):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        if is_presenter:
            self.presenter_connections.add(websocket)
        logger.info(f"New connection. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.presenter_connections:
            self.presenter_connections.remove(websocket)
        logger.info(f"Connection closed. Total: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    async def broadcast(self, message: dict, presenter_only: bool = False):
        """Broadcast a message to all connected clients"""
        connections = self.presenter_connections if presenter_only else self.active_connections
        disconnected = []

        for connection in connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                disconnected.append(connection)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def broadcast_metrics(self, metrics: dict):
        """Broadcast metrics update to all clients"""
        message = {
            "type": "metrics_update",
            "data": metrics
        }
        await self.broadcast(message)

    async def broadcast_state(self, state: dict):
        """Broadcast demo state update to all clients"""
        message = {
            "type": "state_update",
            "data": state
        }
        await self.broadcast(message)

    async def notify_presenter(self, event_type: str, data: dict):
        """Send notification to presenter only"""
        message = {
            "type": event_type,
            "data": data
        }
        await self.broadcast(message, presenter_only=True)

    def get_connection_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)


# Global connection manager instance
connection_manager = ConnectionManager()
