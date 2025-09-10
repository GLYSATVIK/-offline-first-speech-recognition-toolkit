# Architecture

chunkasr is organised as a pipeline of small, independently testable stages.
Data flows left to right; each stage has a narrow, typed interface.

```
audio ─▶ frontend ─▶ (VAD) ─▶ streaming encoder ─▶ decoder ─▶ alignment ─▶ Transcript
        (log-Mel)              (acoustic model)     (CTC/RNN-T)  (timestamps)
```

## Packages

| Package | Responsibility |
| --- | --- |
| `chunkasr.audio` | WAV I/O, resampling, energy+spectral VAD |
| `chunkasr.frontend` | framing, windowing, STFT, Mel banks, CMVN, SpecAugment, deltas |
| `chunkasr.model` | acoustic-model interface, linear reference model, RNN-T transducer, numeric kernels |
| `chunkasr.streaming` | chunker and the streaming encoder with bounded right context |
| `chunkasr.decode` | vocabulary, CTC greedy/beam, RNN-T greedy, n-gram LM |
| `chunkasr.align` | token/word timestamp extraction |
| `chunkasr.metrics` | edit distance, WER/CER, RTF, text normalisation |
| `chunkasr.data` | JSONL manifest tooling |
| `chunkasr.checkpoint` | pickle-free `.npz` import/export |
| `chunkasr.pipeline` | the high-level `Recognizer` |
| `chunkasr.cli` | `transcribe`, `mic`, `score`, `manifest`, `info` |

## Key interfaces

### AcousticModel

Decoders never see a neural network directly — they consume an emission matrix
of shape `(n_frames, vocab_size)`. Any object implementing the
`AcousticModel` protocol (`forward`, `input_dim`, `vocab_size`) is pluggable. The
shipped `LinearAcousticModel` is a single affine projection plus log-softmax; a
PyTorch model implementing the same protocol drops in unchanged.

### StreamingEncoder

The streaming encoder buffers feature frames and emits a frame only once its
right-context lookahead has arrived. This bounds latency to `right_context`
frames and guarantees that streaming and offline decoding produce identical
emissions (see `tests/test_streaming.py`).

### Decoders

- **CTC greedy** — argmax path collapse.
- **CTC prefix beam search** — log-domain beam search tracking blank/non-blank
  prefix probabilities, with an optional `PrefixScorer` for shallow LM fusion.
- **RNN-T greedy** — frame-synchronous greedy transduction with a recurrent
  predictor and feed-forward joiner.

## Design principle

Every stage is a pure function or a small stateful object with an explicit
`reset()`. There is no global state, no implicit device management, and no
network I/O — which is what makes the toolkit reproducible and offline-first.
