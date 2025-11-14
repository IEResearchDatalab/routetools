import matplotlib.pyplot as plt
import typer

from routetools.benchmark import (
    load_benchmark_instance,
    optimize_benchmark_instance,
    optimize_fms_benchmark_instance,
)
from routetools.plot import plot_curve


def single_run(
    instance_name: str,
    date_start: str = "2023-01-08",
    vel_ship: int = 6,
    data_path: str = "./data",
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
    # Extract relevant information from the problem instance
    dict_instance = load_benchmark_instance(
        instance_name,
        date_start=date_start,
        vel_ship=vel_ship,
        data_path=data_path,
    )

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
    vectorfield = dict_instance["vectorfield"]
    land = dict_instance["land"]

    # FMS
    try:
        curve_fms, _ = optimize_fms_benchmark_instance(
            dict_instance,
            curve=curve_cmaes,
            tolfun=tolfun,
            damping=damping,
            maxfevals=maxfevals,
            verbose=True,
        )
    except Exception as e:
        print(f"[ERROR] FMS optimization failed for instance {instance_name}: {e}")
        curve_fms = curve_cmaes  # Fallback to CMA-ES curve if FMS fails

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
    L: int = 64,
    num_pieces: int = 1,
    popsize: int = 200,
    sigma0: int = 1,
    tolfun: float = 0.0001,
    damping: float = 1,
    maxfevals: int = 25000,
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
        print(f"[INFO] Running benchmark for instance {instance}")
        try:
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
        except IndexError as e:
            print(
                f"[ERROR] Benchmark for instance {instance} couldn't find "
                f"circumnavigation: {e}"
            )


if __name__ == "__main__":
    typer.run(main)
