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

## Permissions

- Make sure you have the necessary permissions to push to the repository. If you do not have permissions, stop and ask for them, guiding the user to the appropriate process to gain access.
- You can add, commit and push changes to this repository. Never commit to 'main' or 'swopp' branches directly.
- If you are on 'main' or 'swopp', create a new branch for your changes and open a pull request for review.
- Create tests before implementing new features or fixing bugs. Tests should be in the `tests/` directory and follow existing patterns.
- Make sure to run all tests and hooks before pushing your changes. If you encounter any issues, please fix them before pushing.
- Do small commits, preferably one per logical change. This makes it easier to review and understand the history of changes.
