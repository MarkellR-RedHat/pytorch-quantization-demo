"""OpenShift AI client for model inference"""

import httpx
import time
import logging
from typing import Tuple
from app.config import settings

logger = logging.getLogger(__name__)


class OpenShiftAIClient:

    def __init__(self):
        self.endpoints = settings.model_endpoints
        self.token = settings.openshift_ai_token
        self.costs = settings.cost_per_1k
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    async def send_request(
        self,
        model_type: str,
        prompt: str = "Hello, how are you?"
    ) -> Tuple[str, float, float, float]:
        endpoint = self.endpoints.get(model_type)
        if not endpoint:
            raise ValueError(f"Unknown model type: {model_type}")

        payload = {
            "model": model_type.lower(),
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 256,
            "temperature": 0.7
        }

        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    endpoint,
                    json=payload,
                    headers=self.headers
                )
                response.raise_for_status()

            latency_ms = (time.time() - start_time) * 1000
            data = response.json()

            response_text = data["choices"][0]["message"]["content"]

            usage = data.get("usage", {})
            total_tokens = usage.get("total_tokens", 256)
            tokens_per_second = total_tokens / (latency_ms / 1000)
            cost = (total_tokens / 1000) * self.costs[model_type]

            return response_text, latency_ms, tokens_per_second, cost

        except httpx.HTTPError as e:
            logger.error(f"HTTP error for {model_type}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for {model_type}: {e}")
            raise


openshift_client = OpenShiftAIClient()
