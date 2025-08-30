# Benchmarks

Rough, reproducible micro-benchmarks. These are not rigorous performance claims
— they use the reference (untrained) model on synthetic audio — but they track
relative regressions in the pipeline.

```bash
uv run python benchmarks/benchmark_rtf.py
uv run python benchmarks/benchmark_frontend.py
```
