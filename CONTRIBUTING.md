# Contributing to chunkasr

Thanks for your interest in improving chunkasr! This project values small,
focused changes with tests and clear rationale.

## Development setup

chunkasr uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/ringleadership/chunkasr
cd chunkasr
uv sync --group dev
uv run pytest
```

Install the pre-commit hooks so lint/format run automatically:

```bash
uv run pre-commit install
```

## Before you open a PR

Run the same checks CI runs:

```bash
make check      # ruff lint + format check + mypy + pytest
```

All four must pass on Python 3.10, 3.11 and 3.12.

## Guidelines

- **One concern per commit.** Keep changes small and reviewable.
- **Add tests.** New kernels and decoders should ship with unit tests; DSP code
  should assert on shapes and numerical properties, not just "runs without error".
- **Keep the core dependency-light.** NumPy/SciPy only; PyTorch, `soundfile` and
  `sounddevice` are optional extras and must stay optional (guarded imports).
- **Stay deterministic.** Seed any randomness through `chunkasr.seed`.
- **Type everything.** Public functions need annotations and pass `mypy`.

## Reporting bugs

Open an issue with a minimal reproduction and the output of `chunkasr info`.

## License

By contributing you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
