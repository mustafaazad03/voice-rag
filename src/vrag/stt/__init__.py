"""Provider chain: Sarvam first when its key exists, ElevenLabs as fallback.

Selection is purely a function of which keys are present in the environment:
  SARVAM_API_KEY only        -> sarvam
  ELEVENLABS_API_KEY only    -> elevenlabs
  both                       -> sarvam, then elevenlabs on failure
  neither                    -> voice endpoints return 503, text endpoints still work
"""

from __future__ import annotations

import httpx

from ..config import Settings, get_settings
from ..errors import AudioInvalid, STTProviderError, STTUnavailable
from ..harness.retry import CircuitBreaker, with_retry
from ..models import TranscriptionResult
from ..obs import METRICS, get_logger
from .base import STTProvider
from .elevenlabs import ElevenLabsSTT
from .sarvam import SarvamSTT

log = get_logger("stt")

_BUILDERS = {"sarvam": SarvamSTT, "elevenlabs": ElevenLabsSTT}


class STTRouter:
    """Ordered provider chain with per-provider retries and circuit breakers."""

    def __init__(self, settings: Settings | None = None, client: httpx.AsyncClient | None = None):
        self._s = settings or get_settings()
        self._client = client or httpx.AsyncClient(timeout=self._s.stt_timeout_s)
        self._owns_client = client is None
        self._providers: list[STTProvider] = [
            _BUILDERS[name](self._s, self._client) for name in self._s.stt_providers
        ]
        self._breakers = {p.name: CircuitBreaker(p.name) for p in self._providers}

    @property
    def chain(self) -> list[str]:
        return [p.name for p in self._providers]

    @property
    def available(self) -> bool:
        return bool(self._providers)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def transcribe(
        self,
        audio: bytes,
        filename: str = "audio.wav",
        content_type: str = "",
        language: str | None = None,
    ) -> TranscriptionResult:
        if not self._providers:
            raise STTUnavailable(
                "No speech-to-text provider configured. Set SARVAM_API_KEY or ELEVENLABS_API_KEY.",
                configured=[],
            )
        if not audio:
            raise AudioInvalid("Empty audio payload")
        if len(audio) > self._s.stt_max_audio_bytes:
            raise AudioInvalid(
                "Audio too large", size=len(audio), limit=self._s.stt_max_audio_bytes
            )

        errors: dict[str, str] = {}
        for idx, provider in enumerate(self._providers):
            try:
                result = await with_retry(
                    lambda p=provider: p.transcribe(audio, filename, content_type, language),
                    attempts=self._s.stt_max_attempts,
                    timeout_s=self._s.stt_timeout_s,
                    breaker=self._breakers[provider.name],
                    label=f"stt.{provider.name}",
                )
            except Exception as exc:  # noqa: BLE001 - try the next provider
                errors[provider.name] = str(exc)
                METRICS.inc("stt_provider_failed_total", provider=provider.name)
                log.warning("stt_provider_failed", provider=provider.name, error=str(exc))
                continue

            METRICS.inc("stt_success_total", provider=provider.name)
            METRICS.observe("stt_latency", result.latency_ms)
            return result.model_copy(update={"fallback_used": idx > 0})

        raise STTProviderError("All speech-to-text providers failed", providers=errors)
