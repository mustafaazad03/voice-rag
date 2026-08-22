"""ElevenLabs speech-to-text (fallback provider).

POST https://api.elevenlabs.io/v1/speech-to-text
  header: xi-api-key
  multipart: file, model_id, language_code?
  200: {text, language_code, language_probability, words[]}
Docs: https://elevenlabs.io/docs/api-reference/speech-to-text/convert
"""

from __future__ import annotations

import time

import httpx

from ..config import Settings
from ..errors import STTProviderError
from ..models import TranscriptionResult
from .base import STTProvider, part_mime


class ElevenLabsSTT(STTProvider):
    name = "elevenlabs"

    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._s = settings
        self._client = client

    async def transcribe(
        self, audio: bytes, filename: str, content_type: str, language: str | None = None
    ) -> TranscriptionResult:
        t0 = time.perf_counter()
        data: dict[str, str] = {"model_id": self._s.elevenlabs_model}
        if language and language != "unknown":
            # Sarvam speaks BCP-47 ("hi-IN"), ElevenLabs wants ISO-639 ("hi").
            data["language_code"] = language.split("-")[0]

        try:
            resp = await self._client.post(
                self._s.elevenlabs_stt_url,
                headers={"xi-api-key": self._s.elevenlabs_api_key or ""},
                files={"file": (filename, audio, part_mime(content_type, filename))},
                data=data,
            )
        except httpx.HTTPError as exc:
            raise STTProviderError(
                f"elevenlabs transport error: {type(exc).__name__}", provider="elevenlabs"
            ) from exc

        if resp.status_code >= 400:
            err = STTProviderError(
                f"elevenlabs returned {resp.status_code}",
                provider="elevenlabs",
                status=resp.status_code,
                body=resp.text[:300],
            )
            err.retryable = resp.status_code == 429 or resp.status_code >= 500
            raise err

        payload = resp.json()
        text = (payload.get("text") or "").strip()
        if not text:
            raise STTProviderError("elevenlabs returned an empty transcript", provider="elevenlabs")

        return TranscriptionResult(
            text=text,
            provider=self.name,
            language_code=payload.get("language_code"),
            language_probability=payload.get("language_probability"),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )
