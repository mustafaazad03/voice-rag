"""Error taxonomy. Every failure that crosses a boundary becomes one of these."""

from __future__ import annotations

from typing import Any


class VRagError(Exception):
    """Base. `code` is stable and safe to expose; `details` must never carry secrets."""

    code = "internal_error"
    http_status = 500
    retryable = False

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def envelope(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


class ConfigError(VRagError):
    code = "config_error"
    http_status = 500


class IndexNotReady(VRagError):
    code = "index_not_ready"
    http_status = 503


class STTUnavailable(VRagError):
    """No provider configured at all."""

    code = "stt_unavailable"
    http_status = 503


class STTProviderError(VRagError):
    code = "stt_provider_error"
    http_status = 502
    retryable = True


class STTTimeout(STTProviderError):
    code = "stt_timeout"
    retryable = True


class AudioInvalid(VRagError):
    code = "audio_invalid"
    http_status = 400


class QueryRejected(VRagError):
    """Guardrail refused the input. `reason` says which guardrail."""

    code = "query_rejected"
    http_status = 422


class BudgetExceeded(VRagError):
    code = "budget_exceeded"
    http_status = 504


class CircuitOpen(VRagError):
    code = "circuit_open"
    http_status = 503
    retryable = True


class RateLimited(VRagError):
    code = "rate_limited"
    http_status = 429


class GenerationError(VRagError):
    code = "generation_error"
    http_status = 502
    retryable = True
