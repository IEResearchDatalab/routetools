# CMA-ES Bézier

## Structure

The repository is structured into the following directories:

- `/routetools`: Python source code
- `/tests`: Python code for testing via pytest

Conveniently, a set of workflows via Github Actions are already installed:

- `pre-commit`: run pre-commit hooks
- `pytest`: automatically discover and runs tests in `tests/`

Tools:

- [uv](https://docs.astral.sh/uv/): manage dependencies, Python versions and virtual environments
- [ruff](https://docs.astral.sh/ruff/): lint and format Python code
- [mypy](https://mypy.readthedocs.io/): check types
- [pytest](https://docs.pytest.org/en/): run unit tests
- [pre-commit](https://pre-commit.com/): manage pre-commit hooks
- [prettier](https://prettier.io/): format YAML and Markdown
- [codespell](https://github.com/codespell-project/codespell): check spelling in source code

## Installation

### Application

Install package and pinned dependencies with the [`uv`](https://docs.astral.sh/uv/) package manager:

1. Install `uv`. See instructions for Windows, Linux or MacOS [here](https://docs.astral.sh/uv/getting-started/installation/).

2. Clone repository

3. Install package and dependencies in a virtual environment:

   ```{bash}
   uv sync
   ```

4. Run any command or Python script with `uv run`, for instance:

   ```{bash}
   uv run routetools/cmaes.py
   ```

   Alternatively, you can also activate the virtual env and run the scripts normally:

   ```{bash}
   source .venv/bin/activate
   ```

### Library

Install a specific version of the package with `pip` or `uv pip`:

```{bash}
pip install git+ssh://git@github.com:Weather-Routing-Research/cmaes_bezier_demo.git
```

### Benchmark data

The benchmark data used in the examples and tests is not included in the repository. You can download it from the [weather-routing-benchmarks](https://github.com/Weather-Routing-Research/weather-routing-benchmarks).

Clone the repository outside of this one and point to the data folder when loading a benchmark instance, for example:

```python
from wrr_bench.benchmark import load

dict_instance = load(
    "DEHAM-USNYC",
    date_start="2023-01-08",
    vel_ship=6,
    data_path="../weather-routing-benchmarks/data",
)

print("The problem instance contains the following information:")
print(", ".join(list(dict_instance.keys())))
```

Our advice is to follow the following folder structure:

```
some_folder/
   routetools/  <- this repository
   weather-routing-benchmarks/  <- benchmark data
```

From `routetools` you can then point to the data folder with `../weather-routing-benchmarks/data`.

And to install `weather-routing-benchmarks` as a package in editable mode, run the following from `routetools`:

```bash
uv run pip install -e ../weather-routing-benchmarks
```

## Run examples

There are several examples in the `scripts/` folder. You can run them with `uv run`, for instance:

```bash
uv run scripts/single_run.py
```

If your computer does not have a GPU, you can force JAX to use the CPU with `JAX_PLATFORMS=cpu` before the command. For instance:

```bash
JAX_PLATFORMS=cpu uv run scripts/single_run.py
```

## Setup development environment (Unix)

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and pre-commit hooks:

```{bash}
make install
```

`uv` will automatically create a virtual environment with the specified Python version in `.python-version` and install the dependencies from `uv.lock` (both standard and dev dependencies). It will also install the package in editable mode.

### Adding new dependencies

Add dependencies with:

```{bash}
uv add <PACKAGE>
```

Add dev dependencies with:

```{bash}
uv add --dev <PACKAGE>
```

Remove dependency with:

```{bash}
uv remove <PACKAGE>
```

In all cases `uv` will automatically update the `uv.lock` file and sync the virtual environment. This can also be done manually with:

```{bash}
uv sync
```

### Create a kernel for Jupyter

```{bash}
uv run ipython kernel install --user --name=cmaes
```

### Tools

#### Run pre-commit hooks

Hooks are run on modified files before any commit. To run them manually on all files use:

```{bash}
make hooks
```

#### Run linter and formatter

```{bash}
make ruff
```

#### Run tests

```{bash}
make test
```

#### Run type checker

```{bash}
make mypy
```

## Developer practices (quick guide)

Follow these steps locally before pushing or opening a PR to keep the repository stable and CI-green:

1. Parse the lockfile quickly:

```powershell
python tools/tools_parse_uv_lock.py
```

2. Run pre-commit hooks locally (fast feedback):

```powershell
pre-commit run --all-files
```

3. Run linter and type checks (if you have uv environment):

```powershell
# with uv
uv run ruff check --show-source .
uv run mypy --install-types --non-interactive
# or system-installed tools
ruff check --show-source .
mypy --install-types --non-interactive
```

4. Run the test suite:

```powershell
uv run pytest
```

5. If you need to validate reproducible environment creation, run `uv sync --dev --frozen` on a Linux environment (CI will run this step).

If you are on Windows and the project requires platform-specific packages (for example CUDA-enabled JAX), use WSL2 or a Linux runner for the uv sync step.

### Bypassing pre-commit for small, safe commits

Occasionally you may make a tiny, clearly-safe change (for example: updating a small helper script, fixing a documentation typo, or adding a fallback version file) and want to commit immediately without running the full pre-commit chain locally. In these rare cases you can bypass hooks with:

```powershell
git commit --no-verify -m "chore: small doc or safe fix"
```

Use this sparingly. Always prefer running `pre-commit run --all-files` locally or in your CI workflow. Bypassing hooks should be limited to situations where:

- The change is small and low-risk (docs, generated fallback file).
- You have validated the change manually.
- You plan to run the full checks in CI (or immediately afterwards).

If you are unsure whether a change is safe to commit with `--no-verify`, run the hooks locally first.
