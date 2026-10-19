"""Certify every BERS route exported by revision Task 10.

The Task 10 cache stores one ``.npz`` file per field and seed.  This helper
extracts the 200-by-2 route, runs the interval certificate implemented in
``certify_route.py``, and writes both the route and JSON certificate below
``bers_routes``.  Run it from this directory, for example::

    python certify_bers_routes.py /path/to/task10_synthetic_local_opt/curves
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from certify_route import main as certify_route


FIELDS = ("circular", "fourvortices", "doublegyre", "techy", "swirlys")
SEEDS = range(5)


def main(curve_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []

    for field in FIELDS:
        for seed in SEEDS:
            cache = curve_dir / f"{field}_seed{seed}.npz"
            if not cache.is_file():
                raise FileNotFoundError(cache)

            route_path = output_dir / f"{field}_seed{seed}.npy"
            with np.load(cache) as data:
                route = np.asarray(data["curve"], dtype=float)
            if route.shape != (200, 2):
                raise ValueError(f"{cache}: expected route shape (200, 2), got {route.shape}")
            np.save(route_path, route)

            result = certify_route(field, str(route_path))
            result["seed"] = seed
            certificate_path = route_path.with_name(f"{route_path.stem}_certificate.json")
            certificate_path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
            summary.append(result)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("curve_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("bers_routes"))
    args = parser.parse_args()
    main(args.curve_dir.resolve(), args.output_dir.resolve())
