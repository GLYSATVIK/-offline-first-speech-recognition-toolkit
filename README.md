# chunkasr

[![CI](https://github.com/ringleadership/chunkasr/actions/workflows/ci.yml/badge.svg)](https://github.com/ringleadership/chunkasr/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Streaming, offline-first speech-recognition toolkit.** A compact, readable ASR
stack built on NumPy/SciPy reference kernels — from raw audio to timestamped
text — with an optional PyTorch backend, deterministic seeds and reproducible
experiment configs. No cloud, no telemetry, no surprise downloads.

chunkasr is a *toolkit*, not a model zoo: it ships the plumbing (frontend, VAD,
chunked streaming encoder, CTC/RNN-T decoders, LM fusion, alignment, scoring,
manifests, checkpoints, CLI) and a small reference acoustic model so the whole
pipeline runs end to end and deterministically out of the box.


## Features

- **Audio I/O & resampling** — stdlib-only WAV reading, polyphase resampling via SciPy.
- **Frontend** — framing, windowing, STFT, log-Mel / filterbank features, CMVN, SpecAugment, deltas.
- **VAD** — combined energy + spectral-flatness voice-activity detection with min-duration smoothing.
- **Streaming encoder** — fixed-size chunks with bounded right context; offline and online results agree.
- **Decoding** — CTC greedy, CTC prefix beam search, RNN-T greedy, with shallow n-gram LM fusion.
- **Alignment** — token and word timestamps from the CTC path.
- **Scoring** — WER / CER via edit distance, RTF meter, and matching text normalisation.
- **Data & checkpoints** — JSONL manifest tooling and safe (pickle-free) `.npz` checkpoints.
- **CLIs** — `transcribe`, `mic`, `score`, `manifest`, `info`.


## Installation

```bash
pip install chunkasr                # core (numpy, scipy, pyyaml)
pip install "chunkasr[torch]"       # optional PyTorch backend
pip install "chunkasr[mic]"         # optional live microphone input
```

Or from source with [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/ringleadership/chunkasr
cd chunkasr
uv sync
```


## Quickstart

```python
import numpy as np
import chunkasr

# A seeded reference recogniser — no training or downloads required.
rec = chunkasr.Recognizer.demo()

samples, sr = chunkasr.read_wav("utterance.wav")
result = rec.transcribe(samples, sample_rate=sr)

print(result.text)
print(f"RTF = {result.rtf:.3f}")
for word in result.words:
    print(f"[{word.start:.2f}-{word.end:.2f}] {word.word}")
```

Scoring a hypothesis against a reference:

```python
from chunkasr import wer, cer

score = wer("the quick brown fox", "the quick brown fax")
print(score.as_percent(), score.substitutions)
print(cer("hello", "hallo"))
```


## Command line

```bash
chunkasr transcribe audio1.wav audio2.wav --method ctc_beam --timestamps
chunkasr score --ref refs.txt --hyp hyps.txt
chunkasr manifest stats data/train.jsonl
chunkasr info
```


## Streaming

The streaming encoder consumes fixed-size feature chunks and only emits a frame
once its bounded right-context lookahead has arrived, so latency is capped and
the online output matches the offline result frame-for-frame:

```python
rec = chunkasr.Recognizer.demo().enable_streaming(chunk_size=16, right_context=4)
result = rec.transcribe(samples, sample_rate=sr)
```


## Reproducibility

Every stochastic component draws from an explicit, seedable
`numpy.random.Generator`, and experiments are driven by JSON/YAML
`ExperimentConfig` files that live next to their results. See
[`docs/design-notes.md`](docs/design-notes.md) for the rationale.

## Documentation

- [Architecture](docs/architecture.md)
- [Usage guide](docs/usage.md)
- [Design notes](docs/design-notes.md)
- [API reference](docs/api-reference.md)

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## License

Released under the [MIT License](LICENSE).
