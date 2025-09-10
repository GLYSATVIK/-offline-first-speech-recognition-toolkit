# API reference

A quick reference to the public surface. Import paths shown are the canonical
ones; most are also re-exported from the top-level `chunkasr` namespace.

## Top level (`chunkasr`)

| Symbol | Description |
| --- | --- |
| `Recognizer` | high-level recogniser; `.demo()`, `.transcribe()`, `.enable_streaming()`, `.with_lm()` |
| `Transcript` | result dataclass: `text`, `tokens`, `words`, `rtf` |
| `read_wav` / `write_wav` / `load_audio` | audio I/O |
| `resample` | sample-rate conversion |
| `detect_speech` | VAD returning `SpeechSegment`s |
| `FeatureExtractor` / `LogMelExtractor` | frontend |
| `Vocabulary` | symbol table |
| `ctc_greedy_decode` / `ctc_prefix_beam_search` / `rnnt_greedy_decode` | decoders |
| `NGramLM` | character/word n-gram LM |
| `token_timestamps` / `word_timings` | alignment |
| `wer` / `cer` / `corpus_wer` / `corpus_cer` / `normalize` | scoring |
| `save_checkpoint` / `load_checkpoint` | checkpoints |
| `read_manifest` / `write_manifest` | manifests |
| `set_seed` / `rng` | determinism |

## `chunkasr.config`

- `FeatureConfig`, `VADConfig`, `DecodeConfig`, `ModelConfig`, `ExperimentConfig`
- `load_config(path)`, `save_config(config, path)` — JSON or YAML by extension.

## `chunkasr.frontend`

- `frame_signal(signal, frame_length, frame_shift)`
- `get_window(name, length)` — `hann`, `hamming`, `povey`, `rectangular`
- `stft(...)`, `power_spectrogram(...)`
- `hz_to_mel`, `mel_to_hz`, `mel_filterbank(...)`
- `log_mel_spectrogram(signal, config)`, `LogMelExtractor`
- `CMVN`, `apply_cmvn`, `sliding_window_cmvn`
- `SpecAugment`, `spec_augment`
- `FeatureExtractor`, `add_deltas`

## `chunkasr.model`

- `AcousticModel` (protocol): `forward(features) -> log_probs`, `input_dim`, `vocab_size`
- `LinearAcousticModel(weight, bias)`, `.random(input_dim, vocab_size, seed)`
- `Transducer.random(enc_dim, vocab_size, ...)`, `Predictor`, `Joiner`

## `chunkasr.streaming`

- `iter_chunks(features, chunk_size, right_context)` -> `Chunk`
- `StreamingEncoder(model, chunk_size, right_context)`: `accept`, `flush`, `encode`, `reset`

## `chunkasr.decode`

- `Vocabulary(symbols, blank_id, space_symbol)`; `.ascii_lower()`, `.from_text(text)`
- `ctc_greedy_decode(log_probs, blank_id)`
- `ctc_prefix_beam_search(log_probs, beam_size, blank_id, scorer)`
- `rnnt_greedy_decode(encoder_out, transducer)`
- `NGramLM.train(corpus, order)`, `.train_chars(texts, order)`, `.log_prob(context, token)`
- `CharLMScorer(lm, vocab, weight)`, `NoLM`

## `chunkasr.metrics`

- `levenshtein(ref, hyp) -> EditCounts`
- `wer(ref, hyp) -> WERResult`, `corpus_wer(refs, hyps)`
- `cer(ref, hyp) -> float`, `corpus_cer(refs, hyps)`
- `real_time_factor(compute, audio)`, `RTFMeter`
- `TextNormalizer`, `normalize(text)`

## `chunkasr.data`

- `ManifestEntry(audio_filepath, duration, text, utt_id)`
- `read_manifest(path)`, `write_manifest(path, entries)`
- `filter_by_duration(entries, min, max)`, `manifest_stats(entries)`
