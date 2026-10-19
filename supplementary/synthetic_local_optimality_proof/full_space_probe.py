import sys, numpy as np
from problems import PROBLEMS
from action import evaluate
from family import X_of_s
name=sys.argv[1]; p=PROBLEMS[name]
d=np.load(f"routes/{name}_family.npz")
X=X_of_s(d["xbar"],d["nu"],d["s"])
J,G,H,_=evaluate(p,X)
w=np.linalg.eigvalsh(H)
print(f"{name}: at family optimum J={J:.12f} |grad_full|={np.linalg.norm(G):.3e} full-Hessian eig min={w[0]:.3e} (#neg={np.sum(w<0)})")
Xk=X.copy()
for it in range(12):
    J,G,H,dm=evaluate(p,Xk)
    w=np.linalg.eigvalsh(H)
    seg=np.linalg.norm(np.diff(Xk.reshape(-1,2),axis=0),axis=1)
    print(f"  newton it {it}: J={J:.13f} |g|={np.linalg.norm(G):.3e} lam_min={w[0]:.3e} #neg={np.sum(w<0)} min seg={seg.min():.2e} dist={np.linalg.norm(Xk-X):.2e}")
    if np.linalg.norm(G)<1e-13: break
    Xk=Xk-np.linalg.solve(H,G)
