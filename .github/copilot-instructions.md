# Copilot Instructions for routetools

This repository implements weather-routing optimization with JAX and CMA-ES. Keep changes focused, reproducible, and validated.

Write and reason in English.

## Required Workflow

1. Read relevant source files and matching tests before editing.
2. Make the smallest change that satisfies the request.
3. Add or update tests when behavior changes.
4. After each update, run both commands:
   - `make hooks`
   - `make test`
5. If either command fails, fix the issues and rerun both commands.
6. Do not finalize work until both commands pass.

## Code Conventions

- Use the repository toolchain (`uv` via `make` targets).
- Preserve existing public APIs unless the request explicitly requires a breaking change.
- Follow `ruff` and `pytest` settings in `pyproject.toml`.
- Keep docstrings in NumPy style for public functions.
- Prefer vectorized/JAX-friendly implementations in performance-sensitive code paths.

## Testing Conventions

- Place tests in `tests/` near the relevant domain file.
- For bug fixes, add a regression test first whenever practical.
- Keep tests deterministic and lightweight unless a larger benchmark is explicitly requested.

## Data and Artifacts

- Do not commit large generated outputs.
- Treat `data/` contents as potentially large and optional in local environments.
- Fail with clear error messages when optional datasets are missing.
