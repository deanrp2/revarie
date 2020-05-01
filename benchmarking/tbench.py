import sys
import time
import numpy as np

sys.path.append("../")

from revarie import Revarie


# --- need to do time
# --- need to do memory

nug = 0
sill = 1
rang = 0.5

mu = 0

dom_upper = 1
dom_lower = 0

ptnums = np.linspace(1000,4000, 20).astype(np.int)

times = np.zeros_like(ptnums).astype(np.float64)
trials = np.zeros(8)
for j, pn in enumerate(ptnums):
    print(str(pn) + "/4000")
    x = np.linspace(0,1,pn)
    for i in range(8):
        start = time.time()
        t = Revarie(x,mu,nug,sill,rang,"sph")
        t.mnorm()
        trials[i] = time.time() - start
    times[j] = trials.mean()

with open("1D_bench_2020-5-1.dat","w") as f:
    for pn in ptnums:
        f.write("%i"%pn + ",")
    f.write("\n")
    for t in times:
        f.write("%.8E"%t + ",")

