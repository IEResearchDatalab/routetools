import sys
import numpy as np
from global_search import bers_stage1

name = sys.argv[1]
best = None
for K in (9, 12):
    for sigma0 in (1.0, 2.0, 3.0):
        for seed in range(6):
            curve, fb, ng = bers_stage1(name, K=K, sigma0=sigma0, seed=seed)
            print(f"{name} K={K} sigma0={sigma0} seed={seed}: {fb:.6f}", flush=True)
            if best is None or fb < best[0]:
                best = (fb, curve)
np.save(f"routes/{name}_cmaes.npy", best[1])
print("BEST", name, best[0])
