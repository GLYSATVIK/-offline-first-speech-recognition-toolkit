# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-07-14

### Added
- RNN-T reference transducer (predictor + joiner) and greedy transducer decoding.
- Shallow n-gram LM fusion for the CTC prefix beam search (`with_lm`).
- Token and word timestamp extraction from the CTC path.
- `chunkasr manifest` subcommand (validate / stats / filter).
- Character n-gram language model with add-k smoothing and back-off.

### Changed
- `Recognizer` now measures compute time so every `Transcript` reports its RTF.
- Streaming encoder guarantees online output matches the offline result.

### Fixed
- `read_wav` now raises `AudioError` for missing files instead of `OSError`.

## [0.2.0] - 2026-05-25

### Added
- Streaming encoder with bounded right context and a chunker.
- CTC prefix beam search.
- CMVN (global and sliding-window) and SpecAugment.
- WER/CER scoring with matching text normalisation and an RTF meter.
- JSONL manifest tooling and pickle-free `.npz` checkpoints.

### Changed
- Feature extraction refactored into a composable `FeatureExtractor`.

## [0.1.0] - 2026-04-06

### Added
- Initial release: audio I/O, resampling, log-Mel frontend, energy+spectral VAD.
- Linear reference acoustic model and CTC greedy decoding.
- Project scaffolding, CI across Python 3.10–3.12, and the first test suite.

[Unreleased]: https://github.com/ringleadership/chunkasr/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/ringleadership/chunkasr/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ringleadership/chunkasr/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ringleadership/chunkasr/releases/tag/v0.1.0
