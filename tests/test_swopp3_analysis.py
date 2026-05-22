"""Tests for SWOPP3 analysis script helpers."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt

from routetools.analysis_config import (
    AnalysisPaths,
    _configured_output_dirs,
    _experiment_folder,
)


def _load_swopp3_analysis_module():
    """Load the plotting script directly from scripts/swopp3_analysis.py."""
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "swopp3_analysis.py"
    spec = importlib.util.spec_from_file_location("swopp3_analysis", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {module_path}")

    cartopy = ModuleType("cartopy")
    cartopy_crs = ModuleType("cartopy.crs")
    cartopy_feature = ModuleType("cartopy.feature")
    cartopy_crs.PlateCarree = lambda *args, **kwargs: None
    cartopy_feature.LAND = object()
    cartopy_feature.OCEAN = object()
    cartopy_feature.COASTLINE = object()
    cartopy_feature.BORDERS = object()
    cartopy.crs = cartopy_crs
    cartopy.feature = cartopy_feature
    sys.modules.setdefault("cartopy", cartopy)
    sys.modules.setdefault("cartopy.crs", cartopy_crs)
    sys.modules.setdefault("cartopy.feature", cartopy_feature)
    sys.modules.setdefault(
        "routetools.violations",
        SimpleNamespace(find_team_prefix=lambda *_args, **_kwargs: "IEUniversity-1"),
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_swopp3_analysis = _load_swopp3_analysis_module()


def test_configured_output_dirs_reads_swopp3_profiles(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[swopp3.experiments.no_penalty]
output_dir = "output/swopp3_no_penalty"

[swopp3.experiments.split_penalty]
output_dir = "output/swopp3_split_penalty"
""".strip()
    )

    output_dirs = _configured_output_dirs(config_path)

    assert output_dirs == {
        "no_penalty": "swopp3_no_penalty",
        "split_penalty": "swopp3_split_penalty",
    }


def test_experiment_folder_prefers_existing_legacy_output_when_config_target_missing(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[swopp3.experiments.split_penalty]
output_dir = "output/swopp3_split_penalty"
""".strip()
    )
    (tmp_path / "output" / "swopp3_penalty").mkdir(parents=True)

    paths = AnalysisPaths(
        output_dir=tmp_path / "output",
        figs_dir=tmp_path / "analysis",
        config_path=config_path,
    )

    assert _experiment_folder("penalty", paths) == "swopp3_penalty"


def test_experiment_folder_uses_configured_folder_when_present(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[swopp3.experiments.no_penalty]
output_dir = "output/swopp3_no_penalty"
""".strip()
    )
    (tmp_path / "output" / "swopp3_no_penalty").mkdir(parents=True)

    paths = AnalysisPaths(
        output_dir=tmp_path / "output",
        figs_dir=tmp_path / "analysis",
        config_path=config_path,
    )

    assert _experiment_folder("no_penalty", paths) == "swopp3_no_penalty"


def test_setup_style_uses_transparent_backgrounds() -> None:
    """Plot style should default figure and axes backgrounds to transparent."""
    with mpl.rc_context():
        _swopp3_analysis.setup_style()

        assert mpl.rcParams["figure.facecolor"] == "none"
        assert mpl.rcParams["axes.facecolor"] == "none"
        assert mpl.rcParams["savefig.facecolor"] == "none"
        assert mpl.rcParams["savefig.transparent"] is True


def test_save_figure_outputs_writes_pdf_png_and_removes_stale_tikz(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Figure export helper should emit transparent PDF/PNG and remove stale TikZ."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [1, 0], label="line")

    savefig_calls: list[tuple[Path, dict[str, object]]] = []

    def _fake_savefig(path: str | Path, **kwargs: object) -> None:
        savefig_calls.append((Path(path), kwargs))

    monkeypatch.setattr(fig, "savefig", _fake_savefig)

    out = tmp_path / "figure.pdf"
    out.with_suffix(".tikz").write_text("stale")
    _swopp3_analysis._save_figure_outputs(fig, out, bbox_inches="tight")

    assert [path for path, _ in savefig_calls] == [out, out.with_suffix(".png")]
    assert all(kwargs["transparent"] is True for _, kwargs in savefig_calls)
    assert all(kwargs["bbox_inches"] == "tight" for _, kwargs in savefig_calls)
    assert not out.with_suffix(".tikz").exists()
    assert fig.patch.get_alpha() == 0
    assert ax.get_facecolor()[-1] == 0

    plt.close(fig)


def test_save_figure_outputs_hides_suptitle_and_source_for_png_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """PNG hides suptitle/source while PDF keeps them visible; state is restored."""
    fig, ax = plt.subplots()
    fig.suptitle("Main title")
    ax.set_title("Panel title")
    source_text = fig.text(0.01, -0.01, "Source note")

    visibility_by_suffix: dict[str, tuple[bool, bool]] = {}

    def _fake_savefig(path: str | Path, **kwargs: object) -> None:
        suffix = Path(path).suffix
        visibility_by_suffix[suffix] = (
            fig._suptitle is not None and fig._suptitle.get_visible(),
            source_text.get_visible(),
        )

    monkeypatch.setattr(fig, "savefig", _fake_savefig)

    _swopp3_analysis._save_figure_outputs(fig, tmp_path / "figure.pdf")

    assert visibility_by_suffix[".pdf"] == (True, True)
    assert visibility_by_suffix[".png"] == (False, False)
    assert fig._suptitle is not None and fig._suptitle.get_text() == "Main title"
    assert fig._suptitle is not None and fig._suptitle.get_visible() is True
    assert ax.get_title() == "Panel title"
    assert source_text.get_visible() is True

    plt.close(fig)
