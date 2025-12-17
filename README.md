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

### Git credentials for VCS dependencies

When `uv` installs a package from a git repository (VCS dependency), Git may need credentials to fetch the remote. On non-interactive environments this commonly fails with:

```bash
fatal: could not read Username for 'https://github.com': terminal prompts disabled
```

Use one of the following approaches to make VCS fetches non-interactive.

**Option A: SSH (preferred)**

Generate an SSH key (WSL / Linux):

```bash
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/id_ed25519 -N ""
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Add the printed public key to GitHub (Settings → SSH and GPG keys). Test access:

```bash
ssh -T git@github.com
git ls-remote git@github.com:Weather-Routing-Research/weather-routing-benchmarks.git refs/heads/main
```

Then use an SSH pip URL when adding or declaring the dependency:

```bash
uv add 'git+ssh://git@github.com/Weather-Routing-Research/weather-routing-benchmarks.git@main#egg=wrr_bench'
uv sync
```

If you run `uv` from PowerShell on Windows, ensure the Windows SSH agent is running and the key is loaded (Start-Service ssh-agent; ssh-add $env:USERPROFILE\\.ssh\\id_ed25519), or run `uv` from WSL where the key was created.

**Option B: HTTPS with credentials (fallback)**

Use Git Credential Manager or GitHub CLI to cache credentials so Git won't prompt:

PowerShell (Windows):

```powershell
git config --global credential.helper manager-core
gh auth login --hostname github.com --git-protocol https
```

WSL / Linux (use gh or configure a credential helper that works in your environment):

```bash
gh auth login --hostname github.com --git-protocol https
# or configure `git config --global credential.helper cache` for short-term caching
```

After configuring credentials, retry the `uv add` / `uv sync` command.

### Library

Install a specific version of the package with `pip` or `uv pip`:

```{bash}
pip install git+ssh://git@github.com:Weather-Routing-Research/cmaes_bezier_demo.git
```

### Benchmark data

To be able to run the code, you need to download oceanographic data from [Google Drive](https://drive.google.com/file/d/1jE4adphfGBOWhPETZbNcmh17m6kmOet5/view?usp=sharing). This data is stored in a zip file (15.5 GB) and should be extracted to a folder (24.0 GB).

You can use the following bash script to download the data:

```bash
curl -L -C - \
  -o data.zip \
  'https://drive.usercontent.google.com/download?id=1E52akVR--yPNUHB12vUl2IZH8S9-RuHA&export=download&confirm=t'
unzip -o data.zip -d .
```

The extracted folder should have the following structure:

```
data
├── currents
│   ├── 2023-01-01.nc
│   ├── 2023-01-02.nc
│   ├── ...
│   └── 2023-12-31.nc
├── earth-seas-1km-valid.geo.json
└── earth-seas-2km5-valid.geo.json
```

> Note: This data only includes ocean currents. If you want to include wave data, you need to download the full data folder (121 GB, compressed to 43.5 GB) from [here](https://drive.google.com/file/d/1aWQ8u6kT3v5nUo1YJ5u6F2X1Z3Zk9KXW/view?usp=sharing) instead.

## Run examples

There are several examples in the `scripts/` folder. You can run them with `uv run`, for instance:

```bash
uv run scripts/single_run.py
```

If your computer does not have a GPU, you can force JAX to use the CPU with `JAX_PLATFORMS=cpu` before the command. For instance:

```bash
JAX_PLATFORMS=cpu uv run scripts/single_run.py
```

## Using GPU with JAX

### Quick start (session)

Source the activation script to prefer GPU for JAX in your current shell:

```bash
source scripts/activate_gpu.sh
```

This sets `JAX_PLATFORM_NAME=cuda`, `CUDA_VISIBLE_DEVICES=0`, and `XLA_PYTHON_CLIENT_PREALLOCATE=false` so JAX will pick the local GPU by default.

### Project-wide / tooling

The repository includes a `.env` file with the same defaults; tools that read `.env` (or your CI) can load it. You can also set the variables per-command:

```bash
JAX_PLATFORM_NAME=cuda CUDA_VISIBLE_DEVICES=0 uv run scripts/single_run.py
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
uv run ipython kernel install --user --name=routetools
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
