# Synthetic paper workflow

The two TOML files in this directory are the paper-pinned experiment matrices
consumed by `results.py`:

- `config_land.toml` contains the Four Vortices land experiments.
- `config_noland.toml` contains the five literature vector fields without land.

Run either matrix from the repository root:

```bash
uv run scripts/paper/synthetic/results.py --experiment land
uv run scripts/paper/synthetic/results.py --experiment noland
```

`results.py` resolves the matching config relative to its own location, so the
commands work from any checkout without copying configuration files into the
repository root. Use `--path-config PATH` to run an explicit alternative. The
generated JSON records and aggregate CSV files are written below `output/` by
default; `figures.py` and `tables.py` consume those CSV files.
