"""Task 9 (R1-2) — operating-envelope curvature check of the performance model.

The route-wise audit (Task 8) evaluates the velocity-Hessian of the vessel
performance model only at the conditions actually met along the final routes.
This script evaluates the same 2x2 Hessian over a dense grid of the model's
whole validated input domain (Appendix D, Table "Validated input domain"),
independently of any route or ERA5 file:

* ship speed through water ``v``  in (0, 14.5] m/s,
* true wind speed ``TWS``           in [0, 30]  m/s, from every direction,
* significant wave height ``SWH``   in [0, 10]  m,   from every direction.

Without loss of generality the ship heads east, so the earth-frame velocity is
``(v, 0)`` and every relative wind/wave angle is covered by rotating the wind
and wave directions through 360 degrees.  For every grid point it records the
raw (pre-clamp) power and the eigenvalues of the Hessian of the raw power with
respect to the ship velocity vector, which is the regularity (convexity)
condition on the Lagrangian used by the discrete second-order theory
(Ferraro et al., Thms. 10 and 14).

The result is a numerical map of the admissible operating envelope: the
region of (speed, wind) in which the Lagrangian is strictly convex in velocity
for every wind direction, wave height and wave direction in the grid.  It is
a grid evaluation, not an interval-arithmetic proof.

It needs no ERA5 data and runs on a laptop CPU in about a minute::

    uv run python revision/task9_envelope_hessian.py

Outputs (under ``--output-dir``, default ``output/task9_envelope_hessian``)
--------------------------------------------------------------------------
- ``task9_envelope_grid_summary.csv``: PD fraction per (mode, v, TWS) cell,
  minimised over all wind directions, wave heights and wave directions.
- ``task9_envelope_report.md``: grid, counts and the admissible envelope.
- ``task9_envelope_map.pdf``: PD map in the (v, TWS) plane for both modes.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import typer

from routetools.performance import (
    predict_power_raw_velocity_jax,
    predict_power_velocity_hessian_jax,
)

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parent.parent
KNOT = 0.514444


def _grid(
    n_speed: int, n_tws: int, n_wind_dir: int, n_swh: int, n_wave_dir: int
) -> dict[str, np.ndarray]:
    return {
        "v": np.linspace(14.5 / n_speed, 14.5, n_speed),
        "tws": np.linspace(0.0, 30.0, n_tws),
        "wind_dir": np.linspace(0.0, 360.0, n_wind_dir, endpoint=False),
        "swh": np.linspace(0.0, 10.0, n_swh),
        "wave_dir": np.linspace(0.0, 360.0, n_wave_dir, endpoint=False),
    }


def _evaluator(wps: bool):
    def one(v, tws, wind_dir, swh, wave_dir):
        rad = jnp.deg2rad(wind_dir)
        u10 = tws * jnp.sin(rad)
        v10 = tws * jnp.cos(rad)
        ve = v
        vn = jnp.zeros_like(v)
        raw = predict_power_raw_velocity_jax(u10, v10, swh, wave_dir, ve, vn, wps=wps)
        hess = predict_power_velocity_hessian_jax(
            u10, v10, swh, wave_dir, ve, vn, wps=wps
        )
        hess = 0.5 * (hess + hess.T)
        eig = jnp.linalg.eigvalsh(hess)
        return raw, eig[0], eig[1]

    # vmap over wave direction, swh and wind direction; loop over (v, tws).
    f = jax.vmap(one, in_axes=(None, None, None, None, 0))
    f = jax.vmap(f, in_axes=(None, None, None, 0, None))
    f = jax.vmap(f, in_axes=(None, None, 0, None, None))
    return jax.jit(f)


def main(
    output_dir: str = "output/task9_envelope_hessian",
    n_speed: int = 58,
    n_tws: int = 31,
    n_wind_dir: int = 72,
    n_swh: int = 21,
    n_wave_dir: int = 36,
    relative_tolerance: float = 1.0e-10,
) -> None:
    """Evaluate the velocity-Hessian over the validated input domain."""
    out = Path(output_dir)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    g = _grid(n_speed, n_tws, n_wind_dir, n_swh, n_wave_dir)
    wind_dir = jnp.asarray(g["wind_dir"])
    swh = jnp.asarray(g["swh"])
    wave_dir = jnp.asarray(g["wave_dir"])

    rows: list[dict[str, object]] = []
    totals: dict[str, dict[str, int]] = {}
    for mode, wps in (("noWPS", False), ("WPS", True)):
        fun = _evaluator(wps)
        tot = {"points": 0, "raw_positive": 0, "pd_on_raw_positive": 0}
        for v in g["v"]:
            for tws in g["tws"]:
                raw, lam_min, lam_max = fun(
                    jnp.asarray(v), jnp.asarray(tws), wind_dir, swh, wave_dir
                )
                raw = np.asarray(raw)
                lam_min = np.asarray(lam_min)
                lam_max = np.asarray(lam_max)
                scale = np.maximum(np.abs(lam_max), np.finfo(float).tiny)
                pd = lam_min > relative_tolerance * scale
                positive = raw > 0.0
                n = int(raw.size)
                n_pos = int(positive.sum())
                n_pd_pos = int((pd & positive).sum())
                tot["points"] += n
                tot["raw_positive"] += n_pos
                tot["pd_on_raw_positive"] += n_pd_pos
                rows.append(
                    {
                        "mode": mode,
                        "v_mps": round(float(v), 4),
                        "v_kn": round(float(v) / KNOT, 3),
                        "tws_mps": round(float(tws), 3),
                        "n_points": n,
                        "n_raw_positive": n_pos,
                        "n_pd_on_raw_positive": n_pd_pos,
                        "pd_fraction_all": float(pd.mean()),
                        "pd_fraction_raw_positive": (
                            n_pd_pos / n_pos if n_pos else math.nan
                        ),
                        "min_eigenvalue": float(lam_min.min()),
                        "all_pd": bool(pd.all()),
                    }
                )
        totals[mode] = tot
        print(mode, tot)

    with (out / "task9_envelope_grid_summary.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Admissible envelope: for each TWS, the smallest speed above which every
    # grid point (all directions, wave heights) is PD.
    lines = [
        "# Task 9 — velocity-Hessian over the validated input domain",
        "",
        f"Grid: {n_speed} speeds in (0, 14.5] m/s, {n_tws} wind speeds in"
        f" [0, 30] m/s, {n_wind_dir} wind directions, {n_swh} wave heights in"
        f" [0, 10] m, {n_wave_dir} wave directions.",
        f"PD test: smallest eigenvalue > {relative_tolerance:g} x largest"
        " |eigenvalue| (symmetrised Hessian of raw power).",
        "",
    ]
    for mode in ("noWPS", "WPS"):
        t = totals[mode]
        lines += [
            f"## {mode}",
            "",
            f"- grid points: {t['points']:,}",
            f"- raw power > 0: {t['raw_positive']:,}",
            f"- PD where raw power > 0: {t['pd_on_raw_positive']:,}"
            f" ({100 * t['pd_on_raw_positive'] / max(t['raw_positive'], 1):.3f}%)",
            "",
            "| TWS (m/s) | min speed with PD everywhere (m/s) | (kn) |",
            "|---|---|---|",
        ]
        for tws in g["tws"]:
            cells = [
                r
                for r in rows
                if r["mode"] == mode and r["tws_mps"] == round(float(tws), 3)
            ]
            cells.sort(key=lambda r: r["v_mps"])
            threshold = None
            for i in range(len(cells)):
                if all(c["all_pd"] for c in cells[i:]):
                    threshold = cells[i]["v_mps"]
                    break
            if threshold is None:
                lines.append(f"| {tws:.0f} | none | none |")
            else:
                lines.append(
                    f"| {tws:.0f} | {threshold:.2f} | {threshold / KNOT:.1f} |"
                )
        lines.append("")
    (out / "task9_envelope_report.md").write_text("\n".join(lines) + "\n")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
        for ax, mode in zip(axes, ("noWPS", "WPS"), strict=True):
            sub = [r for r in rows if r["mode"] == mode]
            grid = np.full((len(g["tws"]), len(g["v"])), np.nan)
            for r in sub:
                i = int(np.argmin(np.abs(g["tws"] - r["tws_mps"])))
                j = int(np.argmin(np.abs(g["v"] - r["v_mps"])))
                grid[i, j] = r["pd_fraction_all"]
            im = ax.pcolormesh(
                g["v"] / KNOT,
                g["tws"],
                grid,
                vmin=0,
                vmax=1,
                cmap="viridis",
                shading="nearest",
            )
            ax.set_title("WPS enabled" if mode == "WPS" else "WPS disabled")
            ax.set_xlabel("Speed through water (kn)")
        axes[0].set_ylabel("True wind speed (m/s)")
        fig.colorbar(im, ax=axes, label="Fraction of directions/waves with PD Hessian")
        fig.savefig(out / "task9_envelope_map.pdf", bbox_inches="tight")
    except ImportError:
        pass
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    typer.run(main)
