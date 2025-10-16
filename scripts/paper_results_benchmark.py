import matplotlib.pyplot as plt
import typer
from wrr_bench.benchmark import load

from routetools.benchmark import (
    extract_benchmark_instance,
    optimize_benchmark_instance,
    optimize_fms_benchmark_instance,
)
from routetools.plot import plot_curve


def single_run(
    instance_name: str,
    penalty: float = 10.0,
    K: int = 6,
    L: int = 64,
    num_pieces: int = 1,
    popsize: int = 200,
    sigma0: int = 1,
    tolfun: float = 0.0001,
    damping: float = 1,
    maxfevals: int = 25000,
):
    """Run a single benchmark instance and save the result to output/."""
    dict_instance = load(
        instance_name,
        date_start="2023-01-08",
        vel_ship=6,
        data_path="../weather-routing-benchmarks/data",
    )

    # Extract relevant information from the problem instance
    dict_extracted = extract_benchmark_instance(dict_instance)

    print("The problem instance contains the following information:")
    print(", ".join(list(dict_instance.keys())))

    curve_cmaes, _ = optimize_benchmark_instance(
        dict_instance,
        penalty=penalty,
        K=K,
        L=L,
        num_pieces=num_pieces,
        popsize=popsize,
        sigma0=sigma0,
        tolfun=tolfun,
        damping=damping,
        maxfevals=maxfevals,
    )
    vectorfield = dict_extracted["vectorfield"]
    land = dict_extracted["land"]

    # FMS
    curve_fms, _ = optimize_fms_benchmark_instance(
        dict_instance,
        curve=curve_cmaes,
        tolfun=tolfun,
        damping=damping,
        maxfevals=maxfevals,
        verbose=True,
    )

    # Plot the curve
    plot_curve(
        vectorfield=vectorfield,
        ls_curve=[curve_cmaes, curve_fms],
        ls_name=["CMA-ES", "FMS"],
        land=land,
        gridstep=0.5,
        figsize=(6, 6),
        xlim=(land.xmin, land.xmax),
        ylim=(land.ymin, land.ymax),
    )
    plt.savefig(f"output/benchmark_{instance_name}.jpg", dpi=300)
    plt.close()


def main(
    penalty: float = 10.0,
    K: int = 6,
    L: int = 128,
    num_pieces: int = 1,
    popsize: int = 1000,
    sigma0: int = 3,
    tolfun: float = 1e-5,
    damping: float = 1,
    maxfevals: int = 1000000,
):
    """Run benchmark instances and save the results to output/."""
    ls_instances = [
        "DEHAM-USNYC",
        "USNYC-DEHAM",
        "PAONX-USNYC",
        "USNYC-PAONX",
        "MYKUL-EGHRG",
        "EGHRG-MYKUL",
    ]
    for instance in ls_instances:
        print(f"Running benchmark for instance {instance}")
        single_run(
            instance,
            penalty=penalty,
            K=K,
            L=L,
            num_pieces=num_pieces,
            popsize=popsize,
            sigma0=sigma0,
            tolfun=tolfun,
            damping=damping,
            maxfevals=maxfevals,
        )


if __name__ == "__main__":
    typer.run(main)
