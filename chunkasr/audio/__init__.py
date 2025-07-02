"""Audio input/output, resampling and voice-activity detection."""

from __future__ import annotations

from .io import audio_duration, load_audio, read_wav, write_wav
from .resample import resample
from .vad import EnergyVAD, SpeechSegment, detect_speech

__all__ = [
    "EnergyVAD",
    "SpeechSegment",
    "audio_duration",
    "detect_speech",
    "load_audio",
    "read_wav",
    "resample",
    "write_wav",
]
