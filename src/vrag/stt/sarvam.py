"""Sarvam AI speech-to-text.

POST https://api.sarvam.ai/speech-to-text
  header: api-subscription-key
  multipart: file, model, mode, language_code
  200: {request_id, transcript, language_code, language_probability, timestamps?}
Docs: https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe
"""

from __future__ import annotations

import time

import httpx

from ..config import Settings
from ..errors import STTProviderError
from ..models import TranscriptionResult
from .base import STTProvider, part_mime


class SarvamSTT(STTProvider):
    name = "sarvam"

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._s = settings
        self._client = client

    async def transcribe(
        self, audio: bytes, filename: str, content_type: str, language: str | None = None
    ) -> TranscriptionResult:
        t0 = time.perf_counter()
        data = {
            "model": self._s.sarvam_model,
            "language_code": language or self._s.sarvam_language_code,
        }
        # `mode` only exists on saaras:v3; sending it to other models is a 400.
        if self._s.sarvam_model.startswith("saaras:v3"):
            data["mode"] = self._s.sarvam_mode

        try:
            resp = await self._client.post(
                self._s.sarvam_stt_url,
                headers={"api-subscription-key": self._s.sarvam_api_key or ""},
                files={"file": (filename, audio, part_mime(content_type, filename))},
                data=data,
            )
        except httpx.HTTPError as exc:
            raise STTProviderError(
                f"sarvam transport error: {type(exc).__name__}", provider="sarvam"
            ) from exc

        if resp.status_code >= 400:
            err = STTProviderError(
                f"sarvam returned {resp.status_code}",
                provider="sarvam",
                status=resp.status_code,
                body=resp.text[:300],
            )
            # 4xx other than 429 is our fault: do not retry, do not trip the breaker on retry.
            err.retryable = resp.status_code == 429 or resp.status_code >= 500
            raise err

        payload = resp.json()
        transcript = (payload.get("transcript") or "").strip()
        if not transcript:
            raise STTProviderError("sarvam returned an empty transcript", provider="sarvam")

        return TranscriptionResult(
            text=transcript,
            provider=self.name,
            language_code=payload.get("language_code"),
            language_probability=payload.get("language_probability"),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
