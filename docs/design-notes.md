# Design notes

Some rationale behind the choices that are not obvious from the code.

## Why reference NumPy kernels?

The goal of chunkasr is to be a *readable* and *reproducible* ASR toolkit, not a
state-of-the-art model. Every kernel — framing, STFT, Mel banks, CTC, beam
search, edit distance — is implemented in plain NumPy/SciPy so it can be read
end to end and used as a teaching or research reference. The optional PyTorch
path exists only to show the interface is backend-agnostic; it computes the
identical projection.

## Why a linear reference acoustic model?

A trained acoustic model would require weights, downloads, and a training
pipeline — none of which belong in an offline-first toolkit's core. The linear
model (`features @ W + b` → log-softmax) with seeded random weights lets the
whole decode/align/score/CLI stack run deterministically out of the box. Users
who have a real model implement the `AcousticModel` protocol and plug it in.

## Reproducibility

- No global RNG state in library code: every stochastic component takes a
  `numpy.random.Generator`. `chunkasr.seed.rng(seed)` returns an isolated one.
- Configs are dataclasses that round-trip through JSON/YAML. An experiment is
  fully described by its `ExperimentConfig`, which is written next to results.
- Checkpoints are `.npz` with `allow_pickle=False`: portable and safe.

## Streaming and right context

The streaming encoder models *bounded lookahead*: the output for frame `t` may
depend on frames up to `t + right_context`, but no further. Concretely, the
encoder only emits a frame once that lookahead has arrived; frames near the end
of a stream are flushed with truncated context. Because emission timing is the
only thing that changes, the online result is bit-for-bit identical to the
offline result — a property we assert in the test suite rather than assume.

## CTC prefix beam search

The implementation tracks, per prefix, the log-probabilities of paths ending in
blank (`p_b`) versus a real symbol (`p_nb`). This distinction is what lets CTC
correctly merge repeated labels separated (or not) by blanks. LM fusion is added
as a constant per new-symbol extension; because `log_add(a + c, b + c) = c +
log_add(a, b)`, that constant factors cleanly out of the per-time accumulation,
so it is applied exactly once per emitted symbol.

## Text normalisation lives with scoring

WER/CER are only meaningful if reference and hypothesis are normalised the same
way. Rather than leave normalisation to the caller (and risk mismatched
pipelines), `TextNormalizer` is applied inside the metric functions by default.

## Known limitations

- The reference acoustic model is not trained, so transcripts from `demo()` are
  not meaningful text — the value is in exercising the full pipeline.
- The n-gram LM uses add-k smoothing with simple back-off, not Kneser–Ney.
- SpecAugment omits time warping (masking only), which is the component that
  matters most in practice and is cheapest to keep deterministic.
