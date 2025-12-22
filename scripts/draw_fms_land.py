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
    frames: int = 20,
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
            route = jnp.asarray(data["curve"])

    # Extract relevant information from the problem instance
    dict_instance = load_benchmark_instance(name)
    land: Land = dict_instance["land"]

    fig, ax = plt.subplots(figsize=(6, 6))

    (line,) = ax.plot(route[:, 0], route[:, 1], "r-", marker="o")
    txt_iter = ax.text(0.5, 5.5, "Iteration: 0", fontsize=12, color="black")
    txt_cost = ax.text(0.5, 5.0, "Cost: ?", fontsize=12, color="black")

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
            costfun=land.cost_function,
            verbose=False,
        )
        costs = float(dict_fms["cost"][0])
        line.set_data(route[:, 0], route[:, 1])
        txt_iter.set_text(f"Iteration: {frame * maxfevals}")
        txt_cost.set_text(f"Cost: {costs:.2f}")
        print(f"Frame {frame}: Cost = {costs:.2f}")
        return [line, txt_iter, txt_cost]

    anim = animation.FuncAnimation(fig, animate, frames=frames, blit=True)

    # Save the animation
    anim.save("output/fms.gif", writer="pillow", fps=10)


if __name__ == "__main__":
    typer.run(main)
