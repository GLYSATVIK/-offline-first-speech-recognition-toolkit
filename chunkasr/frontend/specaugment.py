"""SpecAugment: time and frequency masking for feature-space augmentation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import FloatArray


@dataclass
class SpecAugment:
    """Frequency- and time-masking augmentation.

    Deterministic given a seeded generator, so augmented fixtures stay
    reproducible across runs.
    """

    freq_mask_width: int = 15
    time_mask_width: int = 35
    num_freq_masks: int = 2
    num_time_masks: int = 2
    replace_with_mean: bool = True

    def __call__(
        self, features: FloatArray, generator: np.random.Generator | None = None
    ) -> FloatArray:
        gen = generator or np.random.default_rng()
        out = features.astype(np.float32, copy=True)
        fill = float(out.mean()) if self.replace_with_mean else 0.0
        n_frames, n_freq = out.shape
        for _ in range(self.num_freq_masks):
            _apply_mask(out, axis=1, width=self.freq_mask_width, size=n_freq, gen=gen, fill=fill)
        for _ in range(self.num_time_masks):
            _apply_mask(out, axis=0, width=self.time_mask_width, size=n_frames, gen=gen, fill=fill)
        return out


def _apply_mask(
    features: FloatArray,
    axis: int,
    width: int,
    size: int,
    gen: np.random.Generator,
    fill: float,
) -> None:
    """Zero (or mean-fill) a random band of ``features`` along ``axis``."""
    if width <= 0 or size <= 0:
        return
    span = int(gen.integers(0, min(width, size) + 1))
    if span == 0:
        return
    start = int(gen.integers(0, size - span + 1))
    if axis == 0:
        features[start : start + span, :] = fill
    else:
        features[:, start : start + span] = fill


def spec_augment(
    features: FloatArray,
    seed: int | None = None,
    freq_mask_width: int = 15,
    time_mask_width: int = 35,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> FloatArray:
    """Functional wrapper around :class:`SpecAugment`."""
    augmenter = SpecAugment(
        freq_mask_width=freq_mask_width,
        time_mask_width=time_mask_width,
        num_freq_masks=num_freq_masks,
        num_time_masks=num_time_masks,
    )
    return augmenter(features, np.random.default_rng(seed))
