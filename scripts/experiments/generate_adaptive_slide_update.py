#!/usr/bin/env python
"""Generate audited Great Circle and adaptive-protocol slide figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

NAVY = "#000066"
CYAN = "#78D5EF"
CORAL = "#FF8066"
GREY = "#98A2B3"
DARK_GREY = "#4B5563"
TEXT = "#202633"
MID = "#667085"
WHITE = "#FFFFFF"
OCEAN = "#F8FAFC"
LAND = "#E8EAEE"
EXTENT = (-76, 3, 34, 55)

DATES = [f"{d} Jul" for d in range(15, 25)]
GC = np.array([192.576, 193.820, 194.894, 189.325, 181.067,
               167.024, 145.926, 130.169, 121.038, 116.141])
AP_SAVE = np.array([17.021, 33.153, 31.984, 29.552, 22.184,
                    24.491, 6.807, 7.328, -1.492, 8.067])
PI_SAVE = np.array([45.685, 45.693, 40.574, 35.936, 39.451,
                    32.498, 21.118, 28.645, 15.719, 19.434])
REMAINING = np.arange(354, 17, -24)


def projection():
    return ccrs.Orthographic(central_longitude=-36, central_latitude=43)


def style_map(ax):
    ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
    ax.set_facecolor(OCEAN)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=LAND, edgecolor="none", zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="#8C95A3", linewidth=0.8, zorder=2)
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.6,
                 color="#D6DAE0", alpha=0.8, zorder=0)


def baseline_figure(out: Path):
    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor=WHITE)
    ax = fig.add_axes([0.08, 0.14, 0.61, 0.70])
    y = np.arange(len(DATES))[::-1]
    ax.axvline(0, color=DARK_GREY, linewidth=1.5)
    for yi, a, p in zip(y, AP_SAVE, PI_SAVE, strict=True):
        ax.plot([a, p], [yi, yi], color="#D7DAE0", linewidth=4, zorder=1)
    ax.scatter(AP_SAVE, y, s=145, color=CYAN, edgecolor=NAVY, linewidth=0.8, zorder=3)
    ax.scatter(PI_SAVE, y, s=145, color=CORAL, edgecolor=NAVY, linewidth=0.8, zorder=3)
    ax.scatter([AP_SAVE[8]], [y[8]], s=230, facecolor=WHITE, edgecolor=CORAL,
               linewidth=3, zorder=4)
    ax.annotate("23 Jul: a priori is 1.5 MWh worse",
                xy=(AP_SAVE[8], y[8]), xytext=(5, y[8] + 0.65),
                color=CORAL, fontsize=12, weight="bold",
                arrowprops=dict(arrowstyle="-", color=CORAL, lw=1.5))
    ax.set_yticks(y, DATES, fontsize=12, color=TEXT)
    ax.set_xlabel("ERA5 energy saved relative to Great Circle (MWh)", fontsize=13, color=TEXT)
    ax.set_xlim(-5, 50)
    ax.grid(axis="x", color="#E7E9ED", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=11, colors=MID)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CYAN,
               markeredgecolor=NAVY, markersize=10, label="A priori"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CORAL,
               markeredgecolor=NAVY, markersize=10, label="Perfect information"),
    ], loc="lower right", frameon=False, fontsize=11)
    fig.text(0.06, 0.93, "Great Circle is the common ERA5 baseline",
             fontsize=28, weight="bold", color=NAVY, va="top")
    fig.text(0.06, 0.875, "Positive values save propulsion energy under the same reanalysis weather",
             fontsize=12, color=MID, va="top")
    side = fig.add_axes([0.74, 0.20, 0.22, 0.60]); side.axis("off")
    side.text(0, 0.94, "163.2", fontsize=40, weight="bold", color=DARK_GREY, va="top")
    side.text(0, 0.83, "MWh mean Great Circle cost", fontsize=13, color=TEXT, va="top")
    side.text(0, 0.64, "10.1%", fontsize=40, weight="bold", color=CYAN, va="top")
    side.text(0, 0.53, "mean a priori saving", fontsize=13, color=TEXT, va="top")
    side.text(0, 0.34, "19.5%", fontsize=40, weight="bold", color=CORAL, va="top")
    side.text(0, 0.23, "mean perfect-information saving", fontsize=13, color=TEXT, va="top")
    side.text(0, 0.04, "A priori beats GC: 9 of 10", fontsize=14,
              weight="bold", color=NAVY, va="bottom")
    fig.savefig(out / "great_circle_era5_baseline.png", facecolor=WHITE)
    plt.close(fig)


def load_route(routes_json: Path):
    data = json.loads(routes_json.read_text(encoding="utf-8"))
    route = data["routes"][sorted(data["routes"])[0]]
    if "great_circle" in route:
        return np.asarray(route["great_circle"], float)
    from routetools.swopp3 import case_endpoints, great_circle_route

    src, dst = case_endpoints("AO_WPS")
    return np.asarray(great_circle_route(src, dst, n_points=120), float)


def draw_policy(fig, ax, panel, timeline, gc_route, day: int):
    ax.clear(); panel.clear(); timeline.clear()
    style_map(ax)
    ax.plot(gc_route[:, 0], gc_route[:, 1], color=DARK_GREY, linewidth=2,
            linestyle=":", transform=ccrs.PlateCarree(), zorder=3)
    ax.scatter([gc_route[0, 0], gc_route[-1, 0]], [gc_route[0, 1], gc_route[-1, 1]],
               s=75, color=CORAL, edgecolor=WHITE, linewidth=1.5,
               transform=ccrs.PlateCarree(), zorder=5)
    ax.text(gc_route[0, 0] - 1, gc_route[0, 1] + 1.2, "ESSDR", color=NAVY,
            fontsize=11, weight="bold", ha="right", transform=ccrs.PlateCarree())
    ax.text(gc_route[-1, 0] - 1.4, gc_route[-1, 1] - 1.5, "USNYS", color=NAVY,
            fontsize=11, weight="bold", ha="center", transform=ccrs.PlateCarree())
    panel.axis("off")
    date = 15 + day
    panel.text(0, 0.95, f"{date} JUL 00Z", fontsize=30, weight="bold", color=NAVY, va="top")
    panel.text(0, 0.82, "NEW FORECAST RUN", fontsize=13, weight="bold", color=CORAL, va="top")
    panel.text(0, 0.67, f"{REMAINING[day]} h", fontsize=38, weight="bold", color=CYAN, va="top")
    panel.text(0, 0.57, "remaining voyage", fontsize=13, color=TEXT, va="top")
    panel.text(0, 0.38, "Freeze sailed path\nLoad daily IFS\nOptimize remaining suffix",
               fontsize=14, color=TEXT, va="top", linespacing=1.6)
    panel.text(0, 0.07, "PROTOCOL ONLY\nNO ADAPTIVE RESULT", fontsize=12,
               weight="bold", color=CORAL, va="bottom")
    timeline.set_xlim(-0.5, 14.5); timeline.set_ylim(-1, 1); timeline.axis("off")
    timeline.plot(range(15), np.zeros(15), color="#D7DAE0", linewidth=5)
    timeline.scatter(range(15), np.zeros(15), s=80, color=GREY, zorder=2)
    timeline.scatter([day], [0], s=220, color=CORAL, edgecolor=NAVY, linewidth=1.2, zorder=3)
    for idx in (0, 5, 10, 14):
        timeline.text(idx, -0.42, f"{15+idx} Jul", ha="center", fontsize=10, color=MID)
    timeline.text(7, 0.55, "A different deterministic IFS initialization at every decision",
                  ha="center", fontsize=13, weight="bold", color=NAVY)
    fig.suptitle("Adaptive routing uses a new forecast every day", x=0.055, y=0.96,
                 ha="left", fontsize=27, weight="bold", color=NAVY)


def policy_assets(out: Path, routes_json: Path):
    gc_route = load_route(routes_json)
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=WHITE)
    ax = fig.add_axes([0.05, 0.20, 0.67, 0.68], projection=projection())
    panel = fig.add_axes([0.76, 0.23, 0.20, 0.58])
    timeline = fig.add_axes([0.08, 0.06, 0.84, 0.11])
    draw_policy(fig, ax, panel, timeline, gc_route, 0)
    fig.savefig(out / "adaptive_daily_forecast_policy.png", facecolor=WHITE)
    anim = FuncAnimation(fig, lambda frame: draw_policy(fig, ax, panel, timeline, gc_route, frame),
                         frames=15, interval=750, blit=False)
    anim.save(out / "adaptive_daily_forecast_policy.gif", writer=PillowWriter(fps=1.3))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--routes-json", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_figure(args.output_dir)
    policy_assets(args.output_dir, args.routes_json)


if __name__ == "__main__":
    main()
