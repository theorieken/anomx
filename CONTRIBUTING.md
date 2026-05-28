# Contributing

Anomx is early-stage. Contributions should make the core library easier to compose,
test, or integrate into production time-series workflows.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Checks

Run the core checks before opening a pull request:

```bash
pytest
ruff check .
mypy src/anomx
```

Install the release tooling only when building or publishing distributions:

```bash
python -m pip install -e ".[release]"
python -m build
twine check dist/*
```

## Design Principles

- Keep the core package platform-agnostic.
- Put optional integrations behind optional dependencies.
- Use typed dataclasses or protocols for shared contracts.
- Make anomaly outputs inspectable and auditable.
- Prefer clear algorithms and adapters before deep framework coupling.
