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

Follow these steps locally before pushing or opening a PR to keep the
repository stable and CI-green:

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

5. If you need to validate reproducible environment creation, run
   `uv sync --dev --frozen` on a Linux environment (CI will run this step).

If you are on Windows and the project requires platform-specific packages
(for example CUDA-enabled JAX), use WSL2 or a Linux runner for the uv sync step.
