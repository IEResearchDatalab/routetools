import sys
import time
import numpy as np
from global_search import bers_stage1

for name in sys.argv[1:]:
    t0 = time.time()
    curve, fb, ng = bers_stage1(name)
    np.save(f"routes/{name}_cmaes.npy", curve)
    print(f"{name}: CMA-ES cost={fb:.6f} gens={ng} time={time.time()-t0:.1f}s", flush=True)
