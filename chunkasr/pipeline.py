"""The high-level recogniser.

:class:`Recognizer` wires the frontend, acoustic model, decoder and alignment
into a single ``transcribe`` call and is what most users interact with. It also
measures compute time so every result reports its own real-time factor.
"""

from __future__ import annotations

import time

from .align.timestamps import token_timestamps, word_timings
from .audio.resample import resample
from .config import DecodeConfig, FeatureConfig
from .decode.ctc_greedy import ctc_greedy_decode
from .decode.ctc_prefix_beam import PrefixScorer, ctc_prefix_beam_search
from .decode.lm import CharLMScorer, NGramLM, NoLM
from .decode.rnnt_greedy import rnnt_greedy_decode
from .decode.vocab import Vocabulary
from .exceptions import DecodeError
from .frontend.features import FeatureExtractor
from .model.base import AcousticModel
from .model.linear import LinearAcousticModel
from .model.transducer import Transducer
from .results import Transcript
from .streaming.encoder import StreamingEncoder
from .types import FloatArray


class Recognizer:
    """End-to-end streaming speech recogniser."""

    def __init__(
        self,
        model: AcousticModel,
        vocab: Vocabulary,
        feature: FeatureConfig | None = None,
        decode: DecodeConfig | None = None,
        transducer: Transducer | None = None,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.feature_config = feature or FeatureConfig()
        self.extractor = FeatureExtractor(self.feature_config)
        self.decode = decode or DecodeConfig()
        self.transducer = transducer
        self.streaming: StreamingEncoder | None = None
        self.scorer: PrefixScorer = NoLM()

    @classmethod
    def demo(
        cls,
        seed: int = 1234,
        feature: FeatureConfig | None = None,
        decode: DecodeConfig | None = None,
    ) -> Recognizer:
        """Build a fully-wired recogniser with seeded random weights."""
        vocab = Vocabulary.ascii_lower()
        feat = feature or FeatureConfig()
        dim = FeatureExtractor(feat).output_dim
        model = LinearAcousticModel.random(dim, len(vocab), seed=seed)
        transducer = Transducer.random(dim, len(vocab), seed=seed)
        return cls(model, vocab, feature=feat, decode=decode, transducer=transducer)

    def enable_streaming(self, chunk_size: int = 16, right_context: int = 4) -> Recognizer:
        """Route the acoustic model through a :class:`StreamingEncoder`."""
        self.streaming = StreamingEncoder(self.model, chunk_size, right_context)
        return self

    def with_lm(self, lm: NGramLM, weight: float = 0.5) -> Recognizer:
        """Attach a character n-gram LM for shallow-fusion beam search."""
        self.scorer = CharLMScorer(lm, self.vocab, weight)
        self.decode.method = "ctc_beam"
        self.decode.lm_weight = weight
        return self

    def _encode(self, features: FloatArray) -> FloatArray:
        if self.streaming is not None:
            return self.streaming.encode(features)
        return self.model.forward(features)

    def _decode(self, features: FloatArray, log_probs: FloatArray) -> list[int]:
        method = self.decode.method
        if method == "ctc_greedy":
            return ctc_greedy_decode(log_probs, self.decode.blank_id)
        if method == "ctc_beam":
            return ctc_prefix_beam_search(
                log_probs,
                beam_size=self.decode.beam_size,
                blank_id=self.decode.blank_id,
                scorer=self.scorer,
            )
        if method == "rnnt_greedy":
            if self.transducer is None:
                raise DecodeError("rnnt_greedy requires a transducer")
            return rnnt_greedy_decode(features, self.transducer)
        raise DecodeError(f"unknown decode method: {method!r}")

    def transcribe(self, samples: FloatArray, sample_rate: int = 16000) -> Transcript:
        """Transcribe a mono waveform, returning text, timings and timing stats."""
        audio_seconds = samples.shape[0] / float(sample_rate)
        target_sr = self.feature_config.sample_rate
        if sample_rate != target_sr:
            samples = resample(samples, sample_rate, target_sr)

        start = time.perf_counter()
        features = self.extractor(samples)
        log_probs = self._encode(features)
        token_ids = self._decode(features, log_probs)
        compute_seconds = time.perf_counter() - start

        text = self.vocab.decode_ids(token_ids)
        spans = token_timestamps(log_probs, self.vocab, self.decode.blank_id)
        words = word_timings(spans, self.vocab, self.feature_config.frame_shift, target_sr)
        return Transcript(
            text=text,
            tokens=spans,
            words=words,
            audio_seconds=audio_seconds,
            compute_seconds=compute_seconds,
        )
