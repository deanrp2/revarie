from revarie import Variogram
import numpy as np
import time
import datetime
from .output import write
from pathlib import Path

def bench_variogram(path = "."):
    tot_time = time.time()

    ttag = datetime.datetime.now()
    ttag = ttag.strftime("%Y_%m_%d")
    n = 50
    tlimit = 15 #s
    ns = []
    ts = []
    while time.time() - tot_time < tlimit:
        start = time.time()
        ns.append(n)
        x = np.random.uniform(0,1,n)
        f = np.random.uniform(0,1,n)

        Variogram(x,f)

        ts.append(time.time() - start)
        n = int(n*1.2)

    fname = Path("variogram_timing-" + ttag + ".dat")
    write(np.asarray(ns),np.asarray(ts),Path(path) / fname,"Variogram", "1-D")
