# Copilot Instructions for routetools

This repository implements weather-routing optimization with JAX and CMA-ES. Keep changes focused, reproducible, and validated.

Write and reason in English.

## Required Workflow

1. Read relevant source files and matching tests before editing.
2. Make the smallest change that satisfies the request.

## Code Conventions

- Use the repository toolchain (`uv` via `make` targets).
- Preserve existing public APIs unless the request explicitly requires a breaking change.
- Follow `ruff` and `pytest` settings in `pyproject.toml`.
- Keep docstrings in NumPy style for public functions.
- Prefer vectorized/JAX-friendly implementations in performance-sensitive code paths.

## Testing Conventions

- Place tests in `tests/` near the relevant domain file.
- Keep tests deterministic and lightweight unless a larger benchmark is explicitly requested.

## Data and Artifacts

- Do not commit large generated outputs.
- Treat `data/` contents as potentially large and optional in local environments.
- Fail with clear error messages when optional datasets are missing.

## Permissions

- Make sure you have the necessary permissions to push to the repository. If you do not have permissions, stop and ask for them, guiding the user to the appropriate process to gain access.
- You can add, commit and push changes to this repository. Never commit to 'main' or 'swopp' branches directly.
- If you are on 'main' or 'swopp', create a new branch for your changes and open a pull request for review.
- Do small commits, preferably one per logical change. This makes it easier to review and understand the history of changes.
