import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import typer

from routetools.cmaes import optimize
from routetools.vectorfield import vectorfield_fourvortices


def main(gpu: bool = True) -> None:
    """
    Time the optimization algorithm for different population sizes.

    Test for both CPU and GPU.
    """
    if not gpu:
        jax.config.update("jax_platforms", "cpu")  # type: ignore[no-untyped-call]
    # Check if JAX is using the GPU
    print("JAX devices:", jax.devices())

    src = jnp.array([0, 0])
    dst = jnp.array([6, 2])

    list_popsizes = [100, 250, 500, 1000, 2500, 5000]
    list_times = []

    for popsize in list_popsizes:
        start = time.time()
        optimize(
            vectorfield_fourvortices,
            src=src,
            dst=dst,
            travel_stw=1,
            travel_time=None,
            popsize=popsize,
        )
        list_times.append(time.time() - start)

    label = "GPU" if gpu else "CPU"
    plt.figure()
    plt.plot(list_popsizes, list_times, marker="o")
    plt.xlabel("Population size")
    plt.ylabel("Computation time (s)")
    plt.title(f"CMA-ES ({label})")
    plt.savefig(f"output/comptime_vs_popsize_{label}.png")
    plt.close()


if __name__ == "__main__":
    typer.run(main)
