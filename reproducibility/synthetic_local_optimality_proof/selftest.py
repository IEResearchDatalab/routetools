import numpy as np
from jets import IA
from problems import PROBLEMS
from action import evaluate
from family import eval_family
rng=np.random.default_rng(1)
for name, mode in [("fourvortices","family"),("techy","family"),("swirlys","full"),("doublegyre","family"),("circular","full")]:
    p=PROBLEMS[name]
    if mode=="family":
        d=np.load(f"routes/{name}_family.npz"); c=d["s"]; R=1e-5
        ev=lambda Z,i,h=True: eval_family(p,d["xbar"],d["nu"],Z,interval=i,hessian=h)
    else:
        c=np.load(f"routes/{name}.npy")[1:-1].ravel(); R=1e-6
        ev=lambda Z,i,h=True: evaluate(p,Z,interval=i,hessian=h)
    Jb,Gb,Hb,_=ev(IA(c-R,c+R),True)
    worst=0; viol=0
    for k in range(6):
        z=c+R*rng.uniform(-1,1,c.size)*0.999
        J,G,H,_=ev(z,False)
        tolH=1e-9*np.max(np.abs(H)); tolG=1e-9*np.max(np.abs(G))+1e-15
        viol+=np.sum(H<Hb.lo-tolH)+np.sum(H>Hb.hi+tolH)+np.sum(G<Gb.lo-tolG)+np.sum(G>Gb.hi+tolG)+(J<Jb.lo-1e-12)+(J>Jb.hi+1e-12)
    print(f"{name:13s} {mode:6s} box R={R:g}: violations={viol}  max Hessian width={np.max(Hb.width()):.2e}")
