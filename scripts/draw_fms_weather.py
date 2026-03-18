"""Animate FMS refinement on a safe synthetic wind and wave field.

The script starts from a loopy, wiggly great-circle-like baseline between two
fixed endpoints, applies FMS in small optimization chunks, and writes a GIF
showing how the route evolves over iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import typer

from routetools.cost import cost_function_rise
from routetools.fms import optimize_fms
from routetools.weather import DEFAULT_HS_LIMIT, DEFAULT_TWS_LIMIT, evaluate_weather

SAFE_WIND_MAX = DEFAULT_TWS_LIMIT * 0.9
SAFE_WAVE_MAX = DEFAULT_HS_LIMIT * 0.9
DEFAULT_INITIAL_NOISE_SCALE = 0.2


def _rise_cost(
    curve: jnp.ndarray,
    *,
    windfield,
    wavefield,
    travel_time: float,
    time_offset: float = 0.0,
) -> jnp.ndarray:
    return cost_function_rise(
        windfield=windfield,
        curve=curve,
        travel_time=travel_time,
        wavefield=wavefield,
        wps=False,
        time_offset=time_offset,
    )


@dataclass(frozen=True)
class FmsFrame:
    """Single animation frame for the synthetic FMS run."""

    curve: np.ndarray
    cost: float
    max_tws: float
    max_hs: float
    niter: int


def make_noisy_gc_curve(
    src: jnp.ndarray,
    dst: jnp.ndarray,
    *,
    num_points: int,
    seed: int,
    noise_scale: float = DEFAULT_INITIAL_NOISE_SCALE,
    mode_count: int = 3,
) -> jnp.ndarray:
    """Return a baseline with smooth noise plus loop-like perturbations.

    In this planar synthetic setup, the straight segment between ``src`` and
    ``dst`` plays the role of the great-circle route. We add both lateral
    wiggles and a few spin-like turns so FMS has a harder initial route.
    """
    if num_points < 2:
        raise ValueError(f"num_points must be at least 2, got {num_points}")

    tau = jnp.linspace(0.0, 1.0, num_points, dtype=jnp.float32)
    baseline = src[None, :] + tau[:, None] * (dst - src)[None, :]
    if num_points <= 2 or noise_scale <= 0:
        return baseline

    tangent = dst - src
    tangent_norm = jnp.maximum(jnp.linalg.norm(tangent), 1e-6)
    tangent_unit = tangent / tangent_norm
    normal = jnp.array([-tangent[1], tangent[0]], dtype=jnp.float32) / tangent_norm

    interior_tau = tau[1:-1]
    envelope = jnp.sin(jnp.pi * interior_tau) ** 1.2
    key = jax.random.PRNGKey(seed)
    key_coeffs, key_phases, key_wiggles, key_wiggle_phase, key_spins, key_spin_phase = (
        jax.random.split(key, 6)
    )
    mode_ids = jnp.arange(1, mode_count + 1, dtype=jnp.float32)
    coeffs = jax.random.normal(key_coeffs, (mode_count,), dtype=jnp.float32) / mode_ids
    phases = (
        2.0
        * jnp.pi
        * jax.random.uniform(
            key_phases,
            (mode_count,),
            dtype=jnp.float32,
        )
    )
    series = jnp.sum(
        coeffs[:, None]
        * jnp.sin(mode_ids[:, None] * jnp.pi * interior_tau[None, :] + phases[:, None]),
        axis=0,
    )
    scale = noise_scale * float(tangent_norm)
    series = series / jnp.maximum(jnp.max(jnp.abs(series)), 1e-6)

    # Add high-frequency lateral wiggles and two-ish loop turns around baseline.
    wiggle_cycles = jax.random.uniform(
        key_wiggles,
        (),
        minval=7.0,
        maxval=11.0,
        dtype=jnp.float32,
    )
    wiggle_phase = (
        2.0 * jnp.pi * jax.random.uniform(key_wiggle_phase, (), dtype=jnp.float32)
    )
    wiggles = jnp.sin(2.0 * jnp.pi * wiggle_cycles * interior_tau + wiggle_phase)

    spin_turns = 2.0 + 0.35 * jax.random.normal(key_spins, (), dtype=jnp.float32)
    spin_turns = jnp.clip(spin_turns, 1.6, 2.6)
    spin_phase = (
        2.0 * jnp.pi * jax.random.uniform(key_spin_phase, (), dtype=jnp.float32)
    )
    theta = 2.0 * jnp.pi * spin_turns * interior_tau + spin_phase
    spin_radius = 0.55 * scale * envelope

    along_offset = spin_radius * jnp.cos(theta)
    lateral_offset = scale * envelope * (
        0.5 * series + 0.35 * wiggles
    ) + spin_radius * jnp.sin(theta)
    noisy_interior = (
        baseline[1:-1]
        + along_offset[:, None] * tangent_unit[None, :]
        + lateral_offset[:, None] * normal[None, :]
    )
    return baseline.at[1:-1].set(noisy_interior)


def make_safe_synthetic_windfield(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    period: float = 12.0,
):
    """Return a smooth time-varying wind field that stays below SWOPP3 limits."""
    x_mid = 0.5 * (xlim[0] + xlim[1])
    y_mid = 0.5 * (ylim[0] + ylim[1])
    x_span = max(xlim[1] - xlim[0], 1.0)
    y_span = max(ylim[1] - ylim[0], 1.0)

    def windfield(lon, lat, t):
        x = (lon - x_mid) / x_span
        y = (lat - y_mid) / y_span
        phase = 2.0 * jnp.pi * t / period
        speed = (
            SAFE_WIND_MAX
            * (2.0 * jnp.sin(phase + 3.5 * x) + 1.0 * jnp.cos(0.5 * phase - 4.0 * y))
            / 3
        )
        speed = jnp.abs(speed)
        angle = 0.8 * jnp.sin(0.7 * phase + 5.0 * x) + 0.4 * jnp.cos(phase - 4.5 * y)
        return speed * jnp.cos(angle), speed * jnp.sin(angle)

    windfield.is_time_variant = True
    return windfield


def make_safe_synthetic_wavefield(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    period: float = 12.0,
):
    """Return a smooth time-varying wave field that stays below SWOPP3 limits."""
    x_mid = 0.5 * (xlim[0] + xlim[1])
    y_mid = 0.5 * (ylim[0] + ylim[1])
    x_span = max(xlim[1] - xlim[0], 1.0)
    y_span = max(ylim[1] - ylim[0], 1.0)

    def wavefield(lon, lat, t):
        x = (lon - x_mid) / x_span
        y = (lat - y_mid) / y_span
        phase = 2.0 * jnp.pi * t / period
        hs = (
            1.8
            + 0.7 * jnp.sin(0.6 * phase + 4.0 * x)
            + 0.4 * jnp.cos(0.3 * phase - 3.0 * y)
        )
        hs = jnp.clip(hs, 0.0, SAFE_WAVE_MAX)
        mwd = jnp.mod(
            220.0
            + 25.0 * jnp.sin(0.5 * phase + 3.0 * x)
            + 12.0 * jnp.cos(phase - 2.0 * y),
            360.0,
        )
        return hs, mwd

    wavefield.is_time_variant = True
    return wavefield


def sample_field_limits(
    *,
    windfield,
    wavefield,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    travel_time: float,
    grid_size: int = 21,
    time_samples: int = 5,
) -> dict[str, float]:
    """Sample the synthetic fields over the plotting domain."""
    x = jnp.linspace(xlim[0], xlim[1], grid_size)
    y = jnp.linspace(ylim[0], ylim[1], grid_size)
    t = jnp.linspace(0.0, travel_time, time_samples)
    xx, yy, tt = jnp.meshgrid(x, y, t, indexing="ij")
    u, v = windfield(xx, yy, tt)
    hs, _ = wavefield(xx, yy, tt)
    tws = jnp.sqrt(u**2 + v**2)
    return {
        "max_tws": float(jnp.max(tws)),
        "max_hs": float(jnp.max(hs)),
    }


def _route_metrics(
    curve: jnp.ndarray,
    *,
    windfield,
    wavefield,
    travel_time: float,
) -> tuple[float, float, float]:
    curve_batch = curve[None, ...]
    cost = float(
        _rise_cost(
            curve_batch,
            windfield=windfield,
            wavefield=wavefield,
            travel_time=travel_time,
        )[0]
    )
    weather = evaluate_weather(
        curve_batch,
        windfield=windfield,
        wavefield=wavefield,
        travel_time=travel_time,
        spherical_correction=False,
    )
    return cost, float(weather.max_tws[0]), float(weather.max_hs[0])


def simulate_fms_history(
    *,
    src: jnp.ndarray,
    dst: jnp.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    num_points: int = 500,
    frames: int = 80,
    step_fevals: int = 1,
    damping: float = 0.1,
    patience: int = 20,
    travel_time: float = 12.0,
    seed: int = 7,
    initial_noise_scale: float = DEFAULT_INITIAL_NOISE_SCALE,
) -> tuple[list[FmsFrame], dict[str, float]]:
    """Run FMS in small steps and collect route snapshots for animation."""
    vectorfield = make_safe_synthetic_windfield(xlim, ylim, period=travel_time)
    windfield = vectorfield
    wavefield = make_safe_synthetic_wavefield(xlim, ylim, period=travel_time)

    route = make_noisy_gc_curve(
        src,
        dst,
        num_points=num_points,
        seed=seed,
        noise_scale=initial_noise_scale,
    )
    frames_out: list[FmsFrame] = []
    total_niter = 0

    cost0, max_tws0, max_hs0 = _route_metrics(
        route,
        windfield=windfield,
        wavefield=wavefield,
        travel_time=travel_time,
    )
    frames_out.append(
        FmsFrame(
            curve=np.asarray(route),
            cost=cost0,
            max_tws=max_tws0,
            max_hs=max_hs0,
            niter=0,
        )
    )

    for frame in range(frames):
        route_batch, info = optimize_fms(
            vectorfield=vectorfield,
            curve=route,
            windfield=windfield,
            wavefield=wavefield,
            costfun=lambda curve,
            travel_time=None,
            time_offset=0.0,
            **kwargs: _rise_cost(
                curve,
                windfield=windfield,
                wavefield=wavefield,
                travel_time=travel_time,
                time_offset=time_offset,
            ),
            travel_time=travel_time,
            damping=damping,
            patience=patience,
            maxfevals=step_fevals,
            spherical_correction=False,
            enforce_weather_limits=True,
            tws_limit=DEFAULT_TWS_LIMIT,
            hs_limit=DEFAULT_HS_LIMIT,
            verbose=False,
        )
        route = route_batch[0]
        total_niter += int(info["niter"])
        cost_now, max_tws_now, max_hs_now = _route_metrics(
            route,
            windfield=windfield,
            wavefield=wavefield,
            travel_time=travel_time,
        )
        previous_cost = frames_out[-1].cost
        frames_out.append(
            FmsFrame(
                curve=np.asarray(route),
                cost=cost_now,
                max_tws=max_tws_now,
                max_hs=max_hs_now,
                niter=total_niter,
            )
        )
        if cost_now >= previous_cost:
            break
        print(
            f"Frame {frame + 1}/{frames}: "
            f"cost={cost_now:.3f}, max TWS={max_tws_now:.2f}, max Hs={max_hs_now:.2f}"
        )

    return frames_out, sample_field_limits(
        windfield=windfield,
        wavefield=wavefield,
        xlim=xlim,
        ylim=ylim,
        travel_time=travel_time,
    )


def _background_snapshot(
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    sample_time: float,
    grid_size: int,
    windfield,
    wavefield,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = jnp.linspace(xlim[0], xlim[1], grid_size)
    y = jnp.linspace(ylim[0], ylim[1], grid_size)
    xx, yy = jnp.meshgrid(x, y, indexing="xy")
    tt = jnp.full_like(xx, sample_time)
    u, v = windfield(xx, yy, tt)
    hs, _ = wavefield(xx, yy, tt)
    return (
        np.asarray(xx),
        np.asarray(yy),
        np.asarray(u),
        np.asarray(v),
        np.asarray(hs),
    )


def render_animation(
    *,
    history: list[FmsFrame],
    output_path: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    travel_time: float,
    fps: int,
    grid_size: int,
) -> None:
    """Render the collected FMS history to a GIF."""
    windfield = make_safe_synthetic_windfield(xlim, ylim, period=travel_time)
    wavefield = make_safe_synthetic_wavefield(xlim, ylim, period=travel_time)
    xx, yy, u, v, hs = _background_snapshot(
        xlim=xlim,
        ylim=ylim,
        sample_time=0.5 * travel_time,
        grid_size=grid_size,
        windfield=windfield,
        wavefield=wavefield,
    )

    iters = np.array([frame.niter for frame in history], dtype=float)
    costs = np.array([frame.cost for frame in history], dtype=float)
    tws_ratio = np.array([frame.max_tws / DEFAULT_TWS_LIMIT for frame in history])
    hs_ratio = np.array([frame.max_hs / DEFAULT_HS_LIMIT for frame in history])

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0])
    ax_route = fig.add_subplot(grid[:, 0])
    ax_cost = fig.add_subplot(grid[0, 1])
    ax_limits = fig.add_subplot(grid[1, 1])

    contours = ax_route.contourf(xx, yy, hs, levels=12, cmap="Blues", alpha=0.85)
    ax_route.quiver(xx, yy, u, v, color="#444444", alpha=0.7, scale=SAFE_WIND_MAX * 20)
    fig.colorbar(contours, ax=ax_route, label="Hs [m] at mid-passage")

    initial_curve = history[0].curve
    (initial_line,) = ax_route.plot(
        initial_curve[:, 0],
        initial_curve[:, 1],
        linestyle="--",
        color="#888888",
        linewidth=1.5,
        label="Initial route",
    )
    (route_line,) = ax_route.plot(
        initial_curve[:, 0],
        initial_curve[:, 1],
        color="#d1495b",
        linewidth=2.5,
        label="FMS route",
    )
    ax_route.scatter(initial_curve[0, 0], initial_curve[0, 1], color="#2f9e44", s=70)
    ax_route.scatter(initial_curve[-1, 0], initial_curve[-1, 1], color="#f08c00", s=70)
    metrics_text = ax_route.text(
        0.02,
        0.98,
        "",
        transform=ax_route.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    ax_route.set_xlim(*xlim)
    ax_route.set_ylim(*ylim)
    ax_route.set_aspect("equal", adjustable="box")
    ax_route.set_xlabel("x")
    ax_route.set_ylabel("y")
    ax_route.set_title("FMS refinement on safe synthetic weather")
    ax_route.legend(loc="lower right")

    (cost_line,) = ax_cost.plot([], [], color="#1c7ed6", linewidth=2.0)
    (cost_point,) = ax_cost.plot([], [], "o", color="#1c7ed6")
    ax_cost.set_xlim(iters.min(), max(1.0, iters.max()))
    ax_cost.set_ylim(costs.min() * 0.98, costs.max() * 1.02)
    ax_cost.set_xlabel("FMS iterations")
    ax_cost.set_ylabel("Cost")
    ax_cost.set_title("Cost history")
    ax_cost.grid(alpha=0.3)

    (tws_line,) = ax_limits.plot([], [], color="#e67700", linewidth=2.0, label="TWS")
    (hs_line,) = ax_limits.plot([], [], color="#1971c2", linewidth=2.0, label="Hs")
    (tws_point,) = ax_limits.plot([], [], "o", color="#e67700")
    (hs_point,) = ax_limits.plot([], [], "o", color="#1971c2")
    ax_limits.axhline(1.0, color="#c92a2a", linestyle="--", linewidth=1.2)
    ax_limits.set_xlim(iters.min(), max(1.0, iters.max()))
    ax_limits.set_ylim(0.0, max(1.1, float(max(tws_ratio.max(), hs_ratio.max())) * 1.1))
    ax_limits.set_xlabel("FMS iterations")
    ax_limits.set_ylabel("Fraction of limit")
    ax_limits.set_title("Weather margin")
    ax_limits.grid(alpha=0.3)
    ax_limits.legend(loc="upper right")

    def animate(idx: int):
        frame = history[idx]
        route_line.set_data(frame.curve[:, 0], frame.curve[:, 1])
        metrics_text.set_text(
            f"iter={frame.niter}\n"
            f"cost={frame.cost:.3f}\n"
            f"max TWS={frame.max_tws:.2f} m/s\n"
            f"max Hs={frame.max_hs:.2f} m"
        )

        cost_line.set_data(iters[: idx + 1], costs[: idx + 1])
        cost_point.set_data([iters[idx]], [costs[idx]])

        tws_line.set_data(iters[: idx + 1], tws_ratio[: idx + 1])
        hs_line.set_data(iters[: idx + 1], hs_ratio[: idx + 1])
        tws_point.set_data([iters[idx]], [tws_ratio[idx]])
        hs_point.set_data([iters[idx]], [hs_ratio[idx]])

        return [
            initial_line,
            route_line,
            metrics_text,
            cost_line,
            cost_point,
            tws_line,
            hs_line,
            tws_point,
            hs_point,
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=len(history),
        interval=1000 / fps,
        blit=False,
    )
    anim.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def main(
    output_path: Path = Path("output/fms_weather.gif"),
    frames: int = 80,
    step_fevals: int = 50,
    num_points: int = 100,
    damping: float = 0.9,
    patience: int = 20,
    travel_time: float = 12.0,
    seed: int = 14,
    initial_noise_scale: float = DEFAULT_INITIAL_NOISE_SCALE,
    fps: int = 10,
    grid_size: int = 21,
) -> None:
    """Generate a GIF showing FMS refinement on a synthetic safe-weather case."""
    xlim = (0.0, 10.0)
    ylim = (0.0, 6.0)
    src = jnp.array([0.5, 0.75], dtype=jnp.float32)
    dst = jnp.array([9.5, 5.25], dtype=jnp.float32)

    history, field_limits = simulate_fms_history(
        src=src,
        dst=dst,
        xlim=xlim,
        ylim=ylim,
        num_points=num_points,
        frames=frames,
        step_fevals=step_fevals,
        damping=damping,
        patience=patience,
        travel_time=travel_time,
        seed=seed,
        initial_noise_scale=initial_noise_scale,
    )
    render_animation(
        history=history,
        output_path=output_path,
        xlim=xlim,
        ylim=ylim,
        travel_time=travel_time,
        fps=fps,
        grid_size=grid_size,
    )

    typer.echo(f"Saved animation to {output_path}")
    if len(history) - 1 < frames:
        typer.echo(
            "Stopped early: frame cost did not reduce relative to the previous frame."
        )
    typer.echo(
        "Sampled field maxima: "
        f"TWS={field_limits['max_tws']:.2f} m/s, "
        f"Hs={field_limits['max_hs']:.2f} m"
    )
    typer.echo(
        f"Initial cost={history[0].cost:.3f}, final cost={history[-1].cost:.3f}, "
        f"iterations={history[-1].niter}"
    )


if __name__ == "__main__":
    typer.run(main)
