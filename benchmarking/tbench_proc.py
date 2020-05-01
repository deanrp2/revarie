import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so
import datetime

infile = "2D_bench_2020-4-30.dat"
infile = "1D_bench_2020-4-29.dat"

with open(infile, "r") as f:
    d = f.read().splitlines()

n = np.array(d[0].split(",")[:-1]).astype(np.float64)
times = np.array(d[1].split(",")[:-1]).astype(np.float64)

r = so.curve_fit(lambda n, p, k:k*(n)**p, n, times)[0]

print("For k(n)^p: k = {k}   p = {p}".format(k=r[1],p=r[0]))


#%%
#predict runtime
n = 10000
print("Predicted runtime for {n} points is {s}".format(n=n, s =str(datetime.timedelta(seconds = r[1]*n**r[0]))  ))

#plt.loglog(n, times,".",c="k")
