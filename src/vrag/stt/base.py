"""Speech-to-text provider contract."""

from __future__ import annotations

import abc

from ..models import TranscriptionResult


class STTProvider(abc.ABC):
    name: str

    @abc.abstractmethod
    async def transcribe(
        self, audio: bytes, filename: str, content_type: str, language: str | None = None
    ) -> TranscriptionResult: ...


# Extension -> (mime, accepted-by-sarvam). Both providers sniff by content, this is
# only to send a sane multipart part header.
MIME_BY_EXT = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
    "mp4": "audio/mp4",
    "aac": "audio/aac",
    "ogg": "audio/ogg",
    "opus": "audio/ogg",
    "flac": "audio/flac",
    "webm": "audio/webm",
    "aiff": "audio/aiff",
    "amr": "audio/amr",
}


def guess_mime(filename: str, fallback: str = "application/octet-stream") -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return MIME_BY_EXT.get(ext, fallback)


def part_mime(content_type: str, filename: str) -> str:
    """Multipart part header for the upload, safe to send to a provider.

    Browsers report the recorder codec as a parameter — Chrome's MediaRecorder
    gives `audio/webm;codecs=opus`. Sarvam matches its allow-list against the
    whole string, so the parameter turns an accepted `audio/webm` into a 400.
    Strip parameters and keep the bare media type.
    """
    base = content_type.split(";", 1)[0].strip().lower()
    return base or guess_mime(filename)
