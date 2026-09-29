"""Audience side of the arena: anonymous handles, backing a bird, throwing hard prompts"""

import asyncio
import time

from app.websocket import ConnectionManager

ARENA_VARIANTS = ("BF16", "FP8", "INT4_RTN", "INT4_AWQ", "SPEC")
HARD_PROMPT_LABELS = {
    "math": "Multi-step math",
    "logic": "Logic puzzle",
    "code": "Tricky code",
    "long_context": "Long context",
}
MAX_STATE_BYTES = 4096


def handle_for(aid: str) -> str:
    return f"Guest {int(aid, 16) % 900 + 100}"


class ArenaVotes:
    """One vote per anonymous id; voting again moves the vote."""

    def __init__(self):
        self.votes: dict[str, str] = {}

    def back(self, aid: str, variant: str):
        self.votes[aid] = variant

    def summary(self) -> dict:
        counts = {v: 0 for v in ARENA_VARIANTS}
        for variant in self.votes.values():
            counts[variant] += 1
        return {"counts": counts, "total": len(self.votes)}

    def reset(self):
        self.votes.clear()


class ArenaRelay:
    """Relays the presenter's arena state to phones, at most once per min_interval seconds."""

    def __init__(self, manager: ConnectionManager, min_interval: float = 0.5):
        self.manager = manager
        self.min_interval = min_interval
        self.latest: dict | None = None
        self.last_sent = 0.0
        self._task: asyncio.Task | None = None

    async def submit(self, data: dict):
        self.latest = data
        wait = self.min_interval - (time.monotonic() - self.last_sent)
        if wait <= 0:
            await self._flush()
        elif self._task is None or self._task.done():
            self._task = asyncio.create_task(self._delayed(wait))

    async def _delayed(self, wait: float):
        await asyncio.sleep(wait)
        await self._flush()

    async def _flush(self):
        self.last_sent = time.monotonic()
        if self.latest is not None:
            await self.manager.broadcast({"type": "arena_state", "data": self.latest}, target="audience")

    def reset(self):
        self.latest = None
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
