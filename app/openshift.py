"""OpenShift AI client for model inference (vLLM OpenAI-compatible endpoints)"""

import logging
import time

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class VariantNotConfigured(RuntimeError):
    pass


class OpenShiftAIClient:

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def start(self):
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if settings.openshift_ai_token:
                headers["Authorization"] = f"Bearer {settings.openshift_ai_token}"
            self._client = httpx.AsyncClient(timeout=60.0, headers=headers)

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def endpoint(self, model_type: str) -> str:
        endpoint = settings.endpoint_for(model_type)
        if not endpoint:
            raise VariantNotConfigured(
                f"No endpoint configured for {model_type}: set MODEL_{model_type}_ENDPOINT "
                "or enable simulation mode"
            )
        return endpoint

    async def send_request(self, model_type: str, prompt: str) -> tuple[str, float, float, int]:
        """Returns (text, latency ms, completion tokens per second, completion tokens)."""
        endpoint = self.endpoint(model_type)
        await self.start()
        payload = {
            "model": settings.served_name_for(model_type),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0,
            "stream": False,
        }
        start = time.perf_counter()
        response = await self._client.post(endpoint, json=payload)
        latency_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        completion_tokens = int(data.get("usage", {}).get("completion_tokens") or 0)
        tokens_per_second = completion_tokens / (latency_ms / 1000) if latency_ms > 0 else 0.0
        return text, latency_ms, tokens_per_second, completion_tokens


openshift_client = OpenShiftAIClient()
