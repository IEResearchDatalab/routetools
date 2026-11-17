#!/usr/bin/env python
"""Validate canonical January shards and create descriptive data-only figures."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import acquire_weather as A
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

U, V = A.PARAMS["oper"].values()
HS, DIR, PERIOD = A.PARAMS["wave"].values()


def write_csv(path, rows):
    """Write records with a stable header."""
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference(root, valid, field):
    """Read the exact matching reference hour from a daily shard."""
    stamp = np.datetime_as_string(valid, unit="D").replace("-", "")
    year, month = stamp[:4], stamp[4:6]
    path = root / f"era5/v1/atlantic/{year}/{month}/{stamp}_{field}.nc"
    with xr.open_dataset(path) as ds:
        return ds.sel(valid_time=valid).load()


def weighted(a, lat):
    """Average finite ocean samples with cell area weights."""
    weights = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], a.shape)
    mask = np.isfinite(a)
    if not np.any(mask):
        raise ValueError("no eligible finite ocean cells")
    return float(np.sum(np.where(mask, a, 0) * weights) / np.sum(weights[mask]))


def crps(members, truth):
    """Empirical-distribution CRPS, with 50 equally weighted perturbed members."""
    n = members.shape[0]
    ordered = np.sort(members, axis=0)
    coeff = (2 * np.arange(1, n + 1) - n - 1)[:, None, None]
    score = (
        np.mean(np.abs(members - truth), axis=0)
        - np.sum(coeff * ordered, axis=0) / n**2
    )
    good = np.isfinite(truth) & np.isfinite(members).all(axis=0)
    return np.where(good, score, np.nan)


def main():
    """Run the command with the declared schema and validation gates."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root", type=Path, default=Path("/home/SHARED/data/weather-routing")
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/home/SHARED/data/weather-routing/diagnostics/january_2025"),
    )
    args = p.parse_args()
    root = args.root
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((root / "ifs/v1/atlantic/2025/01").glob("*/*/lead_*.nc")) + sorted(
        (root / "era5/v1/atlantic/2025/01").glob("*.nc")
    )
    inventory = []
    stats = []
    for path in files:
        meta = json.loads(path.with_suffix(".json").read_text())
        if not A.completed(path, meta["request"]):
            raise ValueError(f"failed receipt/content verification: {path}")
        if "day" in meta["request"]:
            A.require_final_era5(
                xr.Dataset(attrs=meta["provenance"]["source_attributes"]),
                date.fromisoformat(meta["request"]["day"]),
            )
        meta["final_audit"] = dict(
            validated_at=A.utcnow(),
            validation_script_sha256=A.SCRIPT_SHA256,
            inspector_script_sha256=A.digest(Path(__file__)),
            checks="request, checksum, dimensions, units, physical bounds, "
            "time and member coordinates",
        )
        if "script_hash_capture" not in meta:
            meta["script_hash_capture"] = (
                "file at publication; executing version uncertain because "
                "code was updated during pilot; final_audit records later "
                "validation code"
            )
        A.atomic_json(path.with_suffix(".json"), meta)
        with xr.open_dataset(path) as ds:
            for name in ds.data_vars:
                vals = ds[name].values
                stats.append(
                    dict(
                        path=str(path),
                        variable=name,
                        minimum=float(np.nanmin(vals)),
                        maximum=float(np.nanmax(vals)),
                        finite_fraction=float(np.isfinite(vals).mean()),
                        direction_packing_overshoots=int(
                            np.count_nonzero((vals < 0) | (vals > 360))
                        )
                        if name == DIR
                        else 0,
                    )
                )
        prov = meta["provenance"]
        inventory.append(
            dict(
                path=str(path),
                kind="ifs" if "lead" in meta["request"] else "era5",
                stream=meta["request"].get("stream", "reference"),
                initialization=meta["request"].get("initialization", ""),
                lead=meta["request"].get("lead", ""),
                bytes=meta["bytes"],
                seconds=meta["elapsed_seconds"],
                network_bytes=prov.get("selected_bytes", prov.get("network_bytes", 0)),
                method=prov["method"],
            )
        )
    write_csv(out / "file_inventory.csv", inventory)
    write_csv(out / "physical_checks.csv", stats)
    root_ifs = root / "ifs/v1/atlantic/2025/01"
    scores = []
    control_checks = []
    fans = {}
    mapdata = {}
    rank_rows = []
    for run in ("20250115T0000Z", "20250115T1200Z"):
        for stream, pointstream, field in [
            ("enfo", "oper", "wind"),
            ("waef", "wave", "waves"),
        ]:
            for path in sorted((root_ifs / run / stream).glob("lead_*.nc")):
                lead = int(path.stem.split("_")[1])
                target = root_ifs / run / pointstream / path.name
                if not target.exists():
                    raise ValueError(f"missing paired control {target}")
                with xr.open_dataset(path) as d:
                    ds = d.isel(step=0).load()
                with xr.open_dataset(target) as d:
                    control = d.isel(step=0).load()
                valid = ds.valid_time.values
                ref = reference(root, valid, field)
                for name in ds.data_vars:
                    a = ds[name].sel(number=0).values
                    b = control[name].values
                    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
                        raise ValueError("control missingness differs")
                    maximum = float(np.nanmax(np.abs(a - b)))
                    control_checks.append(
                        dict(
                            run=run,
                            lead=lead,
                            variable=name,
                            max_absolute_difference=maximum,
                        )
                    )
                    if maximum > 1e-5:
                        raise ValueError(
                            f"control differs: {run} {lead} {name} {maximum}"
                        )
                items = (
                    [
                        (
                            "wind_speed",
                            np.hypot(ds[U].values, ds[V].values),
                            np.hypot(ref[U].values, ref[V].values),
                            np.hypot(control[U].values, control[V].values),
                        )
                    ]
                    if field == "wind"
                    else [
                        (
                            "wave_height",
                            ds[HS].values,
                            ref[HS].values,
                            control[HS].values,
                        ),
                        (
                            "wave_period",
                            ds[PERIOD].values,
                            ref[PERIOD].values,
                            control[PERIOD].values,
                        ),
                    ]
                )
                lat = ds.latitude.values
                for name, ensemble, truth, point in items:
                    members = ensemble[1:]
                    # All baselines and probabilistic scores use the same ocean cells.
                    ocean = reference(root, valid, "waves")[HS].values
                    common = (
                        np.isfinite(truth)
                        & np.isfinite(point)
                        & np.isfinite(members).all(axis=0)
                        & np.isfinite(ocean)
                    )
                    truth = np.where(common, truth, np.nan)
                    point = np.where(common, point, np.nan)
                    members = np.where(common[None, :, :], members, np.nan)
                    mean = np.mean(members, axis=0)
                    spread = np.std(members, axis=0, ddof=1)
                    for baseline, estimate in [("control", point), ("mean50", mean)]:
                        error = estimate - truth
                        scores.append(
                            dict(
                                run=run,
                                lead=lead,
                                valid_time=str(valid),
                                quantity=name,
                                baseline=baseline,
                                sample_cells=int(np.isfinite(error).sum()),
                                bias=weighted(error, lat),
                                mae=weighted(np.abs(error), lat),
                                rmse=float(np.sqrt(weighted(error**2, lat))),
                                crps50=weighted(crps(members, truth), lat),
                                rms_spread50=float(np.sqrt(weighted(spread**2, lat))),
                            )
                        )
                    yi = int(np.argmin(np.abs(lat - 45)))
                    xi = int(np.argmin(np.abs(ds.longitude.values + 35)))
                    entry = dict(
                        lead=lead,
                        valid=str(valid),
                        members=members[:, yi, xi],
                        control=float(point[yi, xi]),
                        era5=float(truth[yi, xi]),
                    )
                    fans.setdefault((run, name), []).append(entry)
                    if (
                        run == "20250115T0000Z"
                        and lead == 72
                        and name in ("wave_height", "wind_speed")
                    ):
                        mapdata[name] = (
                            ds.longitude.values,
                            lat,
                            point,
                            mean,
                            truth,
                            spread,
                        )
                    if name == "wave_height":
                        # Allocate rank mass uniformly among ties.
                        good = np.isfinite(truth) & np.isfinite(members).all(axis=0)
                        lower = np.sum(members < truth, axis=0)[good]
                        equal = np.sum(members == truth, axis=0)[good]
                        hist = np.zeros(51)
                        for lo, eq in zip(lower, equal, strict=True):
                            hist[lo : lo + eq + 1] += 1 / (eq + 1)
                        rank_rows.extend(
                            dict(run=run, lead=lead, rank=i, count=float(v))
                            for i, v in enumerate(hist)
                        )
    write_csv(out / "control_equality.csv", control_checks)
    write_csv(out / "scores.csv", scores)
    write_csv(out / "wave_height_ranks.csv", rank_rows)
    for name, (lon, lat, point, mean, truth, spread) in mapdata.items():
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
        common_min = float(np.nanmin([point, mean, truth]))
        common_max = float(np.nanmax([point, mean, truth]))
        arrays = [point, mean, truth, spread, point - truth, mean - truth]
        titles = [
            "Control",
            "Mean of 50 perturbed members",
            "ERA5 reference",
            "Ensemble standard deviation",
            "Control minus ERA5",
            "Mean50 minus ERA5",
        ]
        for i, (ax, a, title) in enumerate(zip(axes.flat, arrays, titles, strict=True)):
            opts = (
                dict(cmap="viridis", vmin=common_min, vmax=common_max)
                if i < 3
                else dict(cmap="viridis")
            )
            if i >= 4:
                lim = max(float(np.nanmax(np.abs(a))), 1e-6)
                opts = dict(cmap="RdBu_r", vmin=-lim, vmax=lim)
            im = ax.pcolormesh(lon, lat, a, shading="auto", **opts)
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set(title=title, xlabel="Longitude", ylabel="Latitude")
        units = "m/s" if name == "wind_speed" else "m"
        fig.suptitle(
            f"{name.replace('_', ' ').title()} ({units}) "
            "· 15 Jan 2025 00 UTC +72 h\n"
            "Descriptive Atlantic pilot; ERA5 is a reference, not observed truth"
        )
        fig.savefig(out / f"{name}_maps.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for row, run in enumerate(("20250115T0000Z", "20250115T1200Z")):
        for col, name in enumerate(("wind_speed", "wave_height")):
            data = sorted(fans[(run, name)], key=lambda r: r["lead"])
            ax = axes[row, col]
            t = [d["lead"] / 24 for d in data]
            members = np.array([d["members"] for d in data])
            ax.plot(t, members, color="#537db1", alpha=0.12, lw=0.7)
            ax.fill_between(
                t,
                np.quantile(members, 0.1, axis=1),
                np.quantile(members, 0.9, axis=1),
                color="#537db1",
                alpha=0.2,
                label="10–90% ensemble range",
            )
            ax.plot(t, [d["control"] for d in data], label="Control", color="#d17b00")
            ax.plot(t, [d["era5"] for d in data], label="ERA5 reference", color="black")
            ax.set(
                title=f"{run} · {name.replace('_', ' ')}",
                xlabel="Forecast lead (days)",
                ylabel="m/s" if name == "wind_speed" else "m",
            )
            ax.legend(fontsize=8)
    fig.suptitle("Offshore point 45°N 35°W · six sampled leads, lines connect samples")
    fig.savefig(out / "ensemble_fans.png", dpi=150)
    plt.close(fig)
    target = np.datetime64("2025-01-18T00:00")
    revisions = []
    for run in sorted(root_ifs.iterdir()):
        if not run.is_dir():
            continue
        init = np.datetime64(
            f"{run.name[:4]}-{run.name[4:6]}-{run.name[6:8]}T{run.name[9:11]}:00"
        )
        lead = int((target - init) / np.timedelta64(1, "h"))
        vals = {}
        for stream in ("oper", "wave"):
            path = run / stream / f"lead_{lead:03d}.nc"
            if not path.exists():
                break
            with xr.open_dataset(path) as ds:
                s = ds.sel(latitude=45, longitude=-35).isel(step=0)
                if stream == "oper":
                    vals["wind_speed"] = float(np.hypot(s[U], s[V]))
                else:
                    vals["wave_height"] = float(s[HS])
        if len(vals) == 2:
            revisions.append(
                dict(
                    initialization=str(init), lead=lead, valid_time=str(target), **vals
                )
            )
    write_csv(out / "fixed_valid_time_revisions.csv", revisions)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, name, field in zip(
        axes, ("wind_speed", "wave_height"), ("wind", "waves"), strict=True
    ):
        ref = reference(root, target, field).sel(latitude=45, longitude=-35)
        truth = float(np.hypot(ref[U], ref[V])) if field == "wind" else float(ref[HS])
        ax.plot(
            [r["initialization"][5:16] for r in revisions],
            [r[name] for r in revisions],
            marker="o",
            label="Successive control forecasts",
        )
        ax.axhline(truth, color="black", ls="--", label="ERA5 reference")
        ax.set(title=name.replace("_", " "), xlabel="Forecast initialization (UTC)")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=8)
    fig.suptitle("Same target: 18 Jan 2025 00 UTC · 45°N 35°W")
    fig.savefig(out / "forecast_revisions.png", dpi=150)
    plt.close(fig)
    summary = dict(
        status="validated",
        files=len(files),
        retained_bytes=sum(r["bytes"] for r in inventory),
        selected_network_bytes=sum(
            r["network_bytes"] for r in inventory if isinstance(r["network_bytes"], int)
        ),
        control_comparisons=len(control_checks),
        max_control_difference=max(
            r["max_absolute_difference"] for r in control_checks
        ),
        sampled_ensemble_leads=[0, 24, 72, 144, 240, 360],
        quantities=[
            "10m wind u/v",
            "significant wave height",
            "mean wave direction",
            "mean wave period",
        ],
        limitations=[
            "two ensemble initializations; six leads, not complete "
            "85-lead ensemble runs",
            "ERA5 shared ancestry and assimilation; no independent buoy"
            " verification yet",
            "area scores describe one small January sample; no "
            "seasonal/calibration claims",
            "no interpolated new forecast fields or route optimization",
            "local reuse timings are conversion, not network download timings",
        ],
    )
    A.atomic_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
