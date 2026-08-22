from __future__ import annotations

import httpx
import pytest

from vrag.config import Settings
from vrag.errors import AudioInvalid, STTProviderError, STTUnavailable
from vrag.stt import STTRouter

AUDIO = b"RIFF....WAVEfmt "


def router_with(handler, **overrides) -> STTRouter:
    settings = Settings(_env_file=None, stt_max_attempts=1, **overrides)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return STTRouter(settings, client)


def test_provider_chain_prefers_sarvam_when_both_keys_exist():
    s = Settings(_env_file=None, sarvam_api_key="a", elevenlabs_api_key="b")
    assert s.stt_providers == ["sarvam", "elevenlabs"]


def test_provider_chain_uses_whichever_key_exists():
    assert Settings(_env_file=None, elevenlabs_api_key="b").stt_providers == ["elevenlabs"]
    assert Settings(_env_file=None, sarvam_api_key="a").stt_providers == ["sarvam"]
    assert Settings(_env_file=None).stt_providers == []


async def test_sarvam_is_used_first_and_sends_its_auth_header():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("api-subscription-key")
        return httpx.Response(200, json={"transcript": "hello world", "language_code": "en-IN"})

    router = router_with(handler, sarvam_api_key="a", elevenlabs_api_key="b")
    result = await router.transcribe(AUDIO, "a.wav", "audio/wav")
    assert result.provider == "sarvam"
    assert result.text == "hello world"
    assert result.fallback_used is False
    assert seen["key"] == "a"
    assert "api.sarvam.ai" in seen["url"]


async def test_elevenlabs_takes_over_when_sarvam_fails():
    def handler(request: httpx.Request) -> httpx.Response:
        if "sarvam" in str(request.url):
            return httpx.Response(500, text="upstream down")
        assert request.headers.get("xi-api-key") == "b"
        return httpx.Response(200, json={"text": "fallback text", "language_code": "en"})

    router = router_with(handler, sarvam_api_key="a", elevenlabs_api_key="b")
    result = await router.transcribe(AUDIO, "a.wav", "audio/wav")
    assert result.provider == "elevenlabs"
    assert result.fallback_used is True


async def test_all_providers_failing_raises():
    router = router_with(lambda r: httpx.Response(500), sarvam_api_key="a", elevenlabs_api_key="b")
    with pytest.raises(STTProviderError):
        await router.transcribe(AUDIO, "a.wav", "audio/wav")


async def test_no_keys_means_no_voice():
    router = router_with(lambda r: httpx.Response(200))
    assert router.available is False
    with pytest.raises(STTUnavailable):
        await router.transcribe(AUDIO, "a.wav", "audio/wav")


async def test_empty_audio_is_rejected_before_any_network_call():
    router = router_with(lambda r: httpx.Response(200), sarvam_api_key="a")
    with pytest.raises(AudioInvalid):
        await router.transcribe(b"", "a.wav", "audio/wav")


async def test_oversized_audio_is_rejected():
    router = router_with(lambda r: httpx.Response(200), sarvam_api_key="a", stt_max_audio_bytes=4)
    with pytest.raises(AudioInvalid):
        await router.transcribe(b"12345", "a.wav", "audio/wav")


async def test_empty_transcript_is_treated_as_a_failure():
    router = router_with(
        lambda r: httpx.Response(200, json={"transcript": "  "}), sarvam_api_key="a"
    )
    with pytest.raises(STTProviderError):
        await router.transcribe(AUDIO, "a.wav", "audio/wav")


async def test_recorder_codec_parameter_is_stripped_from_the_part_header():
    """Chrome sends `audio/webm;codecs=opus`; Sarvam 400s on the parameter."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode("latin-1")
        seen["ct"] = next(
            line.split(":", 1)[1].strip()
            for line in body.splitlines()
            if line.lower().startswith("content-type:") and "audio" in line.lower()
        )
        return httpx.Response(200, json={"transcript": "ok", "language_code": "en-IN"})

    router = router_with(handler, sarvam_api_key="a")
    await router.transcribe(AUDIO, "speech.webm", "audio/webm;codecs=opus")
    assert seen["ct"] == "audio/webm"


async def test_missing_content_type_falls_back_to_the_extension():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content.decode("latin-1")
        return httpx.Response(200, json={"transcript": "ok", "language_code": "en-IN"})

    router = router_with(handler, sarvam_api_key="a")
    await router.transcribe(AUDIO, "speech.wav", "")
    assert "Content-Type: audio/wav" in seen["body"]
