import matplotlib.pyplot as plt
from wrr_bench.benchmark import load

from routetools.benchmark import optimize_benchmark_instance
from routetools.plot import plot_curve

dict_instance = load(
    "DEHAM-USNYC",
    # "PAONX-USNYC",
    # "MYKUL-EGHRG",
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
    maxfevals=1e6,
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
plt.savefig("output/benchmark_result.jpg", dpi=300)
plt.show()
