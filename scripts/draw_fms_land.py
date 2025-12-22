import json
import os

import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import typer
from matplotlib.lines import Line2D

from routetools.benchmark import load_benchmark_instance
from routetools.fms import optimize_fms
from routetools.land import Land
from routetools.vectorfield import vectorfield_zero


def main(
    name: str = "DEHAM-USNYC",
    maxfevals: int = 1,
    damping: float = 0.9,
    frames: int = 50,
    max_distance: int = 100000,  # meters
):
    """Draw the FMS optimization.

    Parameters
    ----------
    maxfevals : int, optional
        The maximum number of iterations, by default 1
    damping : float, optional
        The damping factor, by default 0.1
    frames : int, optional
        The number of frames, by default 20
    """
    # Load the initial route from the circumnavigation JSON file
    path_json_circ = f"output/json_circumnavigation/{name.replace('-', '')}.json"
    if os.path.exists(path_json_circ):
        with open(path_json_circ) as f:
            data = json.load(f)
            route = jnp.asarray(data["curve"])  # (N, 2)

    # Extract relevant information from the problem instance
    dict_instance = load_benchmark_instance(name)
    land: Land = dict_instance["land"]

    # Define the cost function
    def cost_function(vectorfield: None, curve: jnp.ndarray, **kwargs) -> jnp.ndarray:
        return land.cost_function(
            vectorfield, curve, max_distance=max_distance, **kwargs
        )

    # Define X-Y limits as +-2 degrees around the route
    min_x = jnp.min(route[:, 0]) - 2.0
    max_x = jnp.max(route[:, 0]) + 2.0
    min_y = jnp.min(route[:, 1]) - 2.0
    max_y = jnp.max(route[:, 1]) + 2.0

    # Compute figure size to keep proportions
    aspect_ratio = (max_y - min_y) / (max_x - min_x)
    fig, ax = plt.subplots(figsize=(12, 12 * aspect_ratio))

    # Land is a boolean array, so we need to use contourf
    ax.contourf(
        land.x,
        land.y,
        land.array.T,
        levels=[0, 0.5, 1],
        colors=["white", "black", "black"],
        origin="lower",
        zorder=0,
    )

    # Set axis limits and labels
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")

    # Set the aspect ratio to be 1:1
    ax.set_aspect("equal", adjustable="box")

    # Plot the initial route
    (line,) = ax.plot(route[:, 0], route[:, 1], "r-", marker="o")
    # Place texts in axes-relative coordinates so they're always visible
    txt_iter = ax.text(
        0.5,
        0.95,
        "Iteration: 0",
        fontsize=12,
        color="black",
        transform=ax.transAxes,
        ha="center",
    )

    def animate(frame: int) -> list[Line2D]:
        """Animate the FMS optimization.

        Parameters
        ----------
        frame : int
            The frame number

        Returns
        -------
        list[Line2D]
            List of lines to animate
        """
        nonlocal route
        # Run the FMS for one step
        route, dict_fms = optimize_fms(
            vectorfield_zero,
            curve=route,
            damping=damping,
            travel_stw=1,
            maxfevals=maxfevals,
            costfun=cost_function,
            verbose=False,
        )  # (1, N, 2)
        costs = float(dict_fms["cost"][0])
        line.set_data(route[0, :, 0], route[0, :, 1])
        txt_iter.set_text(f"Iteration: {frame * maxfevals}")
        print(f"Frame {frame}: Cost = {costs:.2f}")
        return [line, txt_iter]

    # Disable blitting to ensure full redraw when saving with Pillow
    anim = animation.FuncAnimation(fig, animate, frames=frames, blit=False)

    # Save the animation
    anim.save("output/fms.gif", writer="pillow", fps=10)


if __name__ == "__main__":
    typer.run(main)
