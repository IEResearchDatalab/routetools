# Contributing

This file documents the small workflows we use for maintaining reproducible
environments and typing policy for JAX-based code.

## Updating `uv.lock`

`uv.lock` is an authoritative, committed lockfile. To update it:

1. Create a feature branch: `git checkout -b fix/uv-lock`
2. Regenerate the lockfile locally:

```powershell
irm https://astral.sh/uv/install.sh | iex
uv lock
uv sync --dev
```

3. Run the project's checks locally:

```powershell
uv run pre-commit run --all-files
uv run pytest
```

4. Commit the `uv.lock` change and open a PR. CI will run an additional
   `uv.lock` parse check and the usual test suite.

If `uv sync --frozen` fails in CI after these steps, don't force changes in
`uv.lock` — gather the CI `uv` output and open a PR with the lockfile plus
evidence that tests pass locally.

## JAX typing policy

- Short term: we use narrow `# type: ignore[misc]` only where JAX decorators
  such as `@jit` cause mypy to mark functions as untyped.
- Recommended: add small mypy stubs for `jax`/`jax.numpy` in `typings/` in the
  future to avoid per-site ignores. If you'd like, I can add a minimal `typings`
  folder and update `pyproject.toml`.

## Pre-commit and CI

- Run `uv run pre-commit run --all-files` before opening a PR. CI will run
  the same hooks and fail the PR if checks fail.

## Line endings

We normalize line endings to LF in the repository. `.gitattributes` enforces
this. On Windows, set `git config core.autocrlf false` for a predictable
experience.

Thanks for contributing! If you have questions about the lockfile, open a
PR and tag maintainers.
