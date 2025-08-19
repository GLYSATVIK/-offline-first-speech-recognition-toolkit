from __future__ import annotations

import numpy as np

from chunkasr.config import FeatureConfig
from chunkasr.frontend.cmvn import CMVN, apply_cmvn, sliding_window_cmvn
from chunkasr.frontend.fbank import LogMelExtractor, log_mel_spectrogram
from chunkasr.frontend.features import FeatureExtractor, add_deltas
from chunkasr.frontend.specaugment import SpecAugment, spec_augment
from chunkasr.seed import rng


def _features(tone) -> np.ndarray:
    samples, _ = tone
    return LogMelExtractor(FeatureConfig(num_mel_bins=40))(samples)


def test_logmel_shape(tone) -> None:
    feats = _features(tone)
    assert feats.ndim == 2
    assert feats.shape[1] == 40
    assert np.all(np.isfinite(feats))


def test_log_mel_function_matches_class(tone) -> None:
    samples, _ = tone
    cfg = FeatureConfig(num_mel_bins=40)
    assert np.allclose(log_mel_spectrogram(samples, cfg), LogMelExtractor(cfg)(samples))


def test_cmvn_normalises() -> None:
    feats = rng(0).standard_normal((200, 40)).astype(np.float32)
    out = apply_cmvn(feats)
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(out.std(axis=0), 1.0, atol=1e-2)


def test_cmvn_fit_reuse(tone) -> None:
    feats = _features(tone)
    stats = CMVN.fit(feats)
    assert stats.mean.shape == (40,)
    assert np.all(stats.std > 0.0)


def test_sliding_window_cmvn_shape(tone) -> None:
    feats = _features(tone)
    out = sliding_window_cmvn(feats, window=50)
    assert out.shape == feats.shape


def test_specaugment_determinism(tone) -> None:
    feats = _features(tone)
    a = SpecAugment()(feats, rng(0))
    b = SpecAugment()(feats, rng(0))
    assert np.array_equal(a, b)
    assert a.shape == feats.shape


def test_spec_augment_function(tone) -> None:
    feats = _features(tone)
    out = spec_augment(feats, seed=1)
    assert out.shape == feats.shape


def test_feature_extractor_deltas(tone) -> None:
    samples, _ = tone
    base = FeatureExtractor(FeatureConfig(num_mel_bins=40), delta_order=0)
    with_delta = FeatureExtractor(FeatureConfig(num_mel_bins=40), delta_order=2)
    assert with_delta.output_dim == base.output_dim * 3
    feats = with_delta(samples)
    assert feats.shape[1] == 120


def test_add_deltas(tone) -> None:
    feats = _features(tone)
    out = add_deltas(feats, order=1)
    assert out.shape[1] == feats.shape[1] * 2
