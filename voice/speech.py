"""Voice interface: speech-to-text and text-to-speech.

Planned addition, wired but optional — the orchestrator never depends on
this module succeeding. If the underlying libraries aren't installed, or
there's no microphone/network, every function here degrades to a no-op or
raises a clearly-labelled error that the caller can catch and fall back to
typed text.

STT: Whisper via the Groq API (fast, free tier) is tried first; the
browser's own Web Speech API is the intended fallback in the web UI (no
server code needed for that path). TTS: gTTS is tried first; the browser's
speech synthesis API is the intended fallback in the web UI.
"""
import os

import config


class VoiceUnavailableError(Exception):
    pass


def transcribe_audio(audio_file_path: str, language: str = "en") -> str:
    """Transcribe an audio file to text using Whisper on Groq.

    `language` should be one of config.SUPPORTED_LANGUAGES keys ("en",
    "hi", "kn"). Raises VoiceUnavailableError if no Groq key is configured
    or the request fails, so callers can fall back to a text prompt.
    """
    if not config.GROQ_API_KEY:
        raise VoiceUnavailableError("No GROQ_API_KEY configured for speech-to-text.")
    if not os.path.exists(audio_file_path):
        raise VoiceUnavailableError(f"Audio file not found: {audio_file_path}")

    import requests

    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
    try:
        with open(audio_file_path, "rb") as f:
            files = {"file": f}
            data = {"model": "whisper-large-v3", "language": language}
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=config.REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json().get("text", "").strip()
    except Exception as e:
        raise VoiceUnavailableError(f"Transcription failed: {e}") from e


def synthesize_speech(text: str, language: str = "en", output_path: str = "reply.mp3") -> str:
    """Convert text to a spoken .mp3 file using gTTS.

    Returns the output file path. Raises VoiceUnavailableError if gTTS
    isn't installed or synthesis fails, so callers can fall back to
    displaying text only.
    """
    try:
        from gtts import gTTS
    except ImportError as e:
        raise VoiceUnavailableError("gTTS is not installed; falling back to text output.") from e

    try:
        tts = gTTS(text=text, lang=language)
        tts.save(output_path)
        return output_path
    except Exception as e:
        raise VoiceUnavailableError(f"Speech synthesis failed: {e}") from e


def speak_or_fallback(text: str, language: str = "en"):
    """Best-effort speech: try TTS, otherwise just return the text.

    Convenience function for the CLI/dashboard demo path so calling code
    never has to branch on whether voice is available.
    """
    try:
        path = synthesize_speech(text, language)
        return {"mode": "audio", "path": path}
    except VoiceUnavailableError:
        return {"mode": "text", "text": text}
