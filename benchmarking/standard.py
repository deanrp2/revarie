from revarie import Revarie
from revarie import Variogram
from revarie import fvariogram
import numpy as np
import time
import datetime
from .output import write
from pathlib import Path

def bench_variogram(tlimit = 15, path = "."):
    tot_time = time.time()

    n = 50
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

    t = 0
    fname = Path("variogram_timing" + str(t).zfill(3) + ".dat")
    while (Path(path) / fname).is_file():
        t += 1
        fname = Path("variogram_timing" + str(t).zfill(3) + ".dat")
    write(np.asarray(ns),np.asarray(ts),Path(path) / fname,"Variogram", "1-D")

def bench_revarie(tlimit = 15, path = "."):
    tot_time = time.time()

    nug = 0
    sill = 1
    rang = 0.3
    v = fvariogram("func", "sph", [nug, sill, rang])

    mu = 1

    n = 50
    ns = []
    ts = []
    while time.time() - tot_time < tlimit:
        start = time.time()
        ns.append(n)
        x = np.random.uniform(0,1,n)

        Revarie(x, mu, sill, v).genf()

        ts.append(time.time() - start)
        n = int(n*1.2)

    t = 0
    fname = Path("revarie_timing" + str(t).zfill(3) + ".dat")
    while (Path(path) / fname).is_file():
        t += 1
        fname = Path("revarie_timing" + str(t).zfill(3) + ".dat")
    write(np.asarray(ns),np.asarray(ts),Path(path) / fname,"Revarie", "1-D")

def suite(tlimit = 30, path = "."):
    bench_variogram(tlimit/2, path = path)
    bench_revarie(tlimit/2, path = path)






