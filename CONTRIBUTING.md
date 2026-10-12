# Contributing

Thanks for helping improve qddot. Before making changes, read [`AGENTS.md`](AGENTS.md), which defines the engineering contract and current API boundaries.

## Development setup

Use Python 3.12 and uv:

```sh
uv sync
uv run qddot --help
```

Do not add or upgrade dependencies or edit `pyproject.toml` without coordinating with the maintainers. Keep episode data in a temporary project directory while developing.

## Changes

- Keep MuJoCo in-process and DuckDB embedded; qddot does not require a service.
- Preserve deterministic behavior for a fixed task, controller, seed, perturbation, MuJoCo version, and platform. Seed random generators explicitly.
- Keep CLI output concise and machine output available through `--json` where supported.
- Add deterministic, fast tests for behavior changes. Run `uv run pytest -q` and `uv run ruff check` before submitting.
- Update the docs when public APIs, CLI flags, file formats, or limitations change.
- Do not commit generated `.qddot/` runs or private benchmark fixtures.

Open an issue or discussion for larger API changes before building on an unstable interface.
