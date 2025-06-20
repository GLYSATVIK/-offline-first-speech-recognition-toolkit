"""chunkasr — a streaming, offline-first speech-recognition toolkit.

The public API is re-exported here so common entry points are one import away::

    import chunkasr
    rec = chunkasr.Recognizer.demo()
    result = rec.transcribe(samples, sample_rate=16000)
    print(result.text, result.rtf)
"""

from __future__ import annotations

from ._version import __version__
from .align import token_timestamps, word_timings
from .audio import detect_speech, load_audio, read_wav, resample, write_wav
from .checkpoint import load_checkpoint, save_checkpoint
from .config import ExperimentConfig, load_config, save_config
from .data import read_manifest, write_manifest
from .decode import (
    NGramLM,
    Vocabulary,
    ctc_greedy_decode,
    ctc_prefix_beam_search,
    rnnt_greedy_decode,
)
from .frontend import FeatureExtractor, LogMelExtractor
from .metrics import cer, corpus_cer, corpus_wer, normalize, wer
from .model import LinearAcousticModel, Transducer
from .pipeline import Recognizer
from .results import Transcript
from .seed import rng, set_seed

__all__ = [
    "ExperimentConfig",
    "FeatureExtractor",
    "LinearAcousticModel",
    "LogMelExtractor",
    "NGramLM",
    "Recognizer",
    "Transcript",
    "Transducer",
    "Vocabulary",
    "__version__",
    "cer",
    "corpus_cer",
    "corpus_wer",
    "ctc_greedy_decode",
    "ctc_prefix_beam_search",
    "detect_speech",
    "load_audio",
    "load_checkpoint",
    "load_config",
    "normalize",
    "read_manifest",
    "read_wav",
    "resample",
    "rng",
    "rnnt_greedy_decode",
    "save_checkpoint",
    "save_config",
    "set_seed",
    "token_timestamps",
    "wer",
    "word_timings",
    "write_manifest",
    "write_wav",
]
