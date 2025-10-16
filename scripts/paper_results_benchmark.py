import matplotlib.pyplot as plt
import typer
from wrr_bench.benchmark import load

from routetools.benchmark import optimize_benchmark_instance
from routetools.plot import plot_curve


def single_run(instance_name: str):
    """Run a single benchmark instance and save the result to output/."""
    dict_instance = load(
        instance_name,
        date_start="2023-01-08",
        vel_ship=6,
        data_path="../weather-routing-benchmarks/data",
    )

    print("The problem instance contains the following information:")
    print(", ".join(list(dict_instance.keys())))

    curve_opt, info = optimize_benchmark_instance(
        dict_instance,
        K=6,
        L=128,
        penalty=10,
        num_pieces=1,
        maxfevals=1000000,
        tolfun=1e-5,
        popsize=1000,
        sigma0=3,
    )
    vectorfield = info["vectorfield"]
    land = info["land"]

    # Plot the curve
    plot_curve(
        vectorfield=vectorfield,
        ls_curve=[curve_opt],
        ls_name=["Optimized"],
        land=land,
        gridstep=0.5,
        figsize=(6, 6),
        xlim=(land.xmin, land.xmax),
        ylim=(land.ymin, land.ymax),
    )
    plt.savefig(f"output/benchmark_{instance_name}.jpg", dpi=300)
    plt.close()


def main():
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
        single_run(instance)


if __name__ == "__main__":
    typer.run(main)
