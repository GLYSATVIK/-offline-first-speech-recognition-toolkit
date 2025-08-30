# Examples

Small, self-contained scripts that exercise chunkasr end to end. None of them
download anything or need a trained model — the reference `Recognizer.demo()`
and synthesised audio keep everything offline and deterministic.

Run any of them with:

```bash
uv run python examples/<script>.py
```

| Script | What it shows |
| --- | --- |
| `transcribe_wav.py` | synthesise a tone, write a WAV, transcribe it, print timings |
| `streaming_demo.py` | streaming vs offline encoding produce identical emissions |
| `vad_segments.py` | detect voiced regions with the energy+spectral VAD |
| `score_wer.py` | compute WER/CER between reference and hypothesis lists |
| `lm_beam_search.py` | train a character n-gram LM and decode with shallow fusion |
| `build_manifest.py` | build and summarise a JSONL dataset manifest |
