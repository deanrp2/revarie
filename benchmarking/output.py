import numpy as np
from scipy.optimize import curve_fit
import platform, subprocess
from datetime import datetime, date

def get_processor_info():
    if platform.system() == "Windows":
        r = platform.processor()
    elif platform.system() == "Linux":
        command = "lscpu"
        r = subprocess.check_output(command, shell=True).strip()
    return r.decode("utf-8")


def write(n, t, fname, typ):
    """
    typ should be "variogram" or "revarie"
    """
    out = open(fname, "w")
    #Write CPU information
    out.write("% --- CPU info\n")
    out.write(get_processor_info() + "\n\n")

    #Write testing information
    out.write("% --- Test Info\n")
    out.write("Test Date/Time:".ljust(24) + str(datetime.now()) + "\n")
    out.write("Test Type:".ljust(24) + typ + "\n")
    mn = "%.2E"%n.min()
    mx = "%.2E"%n.max()
    out.write("Nrange:".ljust(24) + mn + "-" + mx + "\n\n")

    #Report fit to exponential of form t=k*n^p
    out.write("% --- Results Summary\n")
    out.write("Total Runtime:".ljust(24) + str(t.sum()) + " s\n")
    a_est = np.log(t[0]/t[-1])/np.log(n[0]/n[-1])
    k_est = t[-1]/n[-1]**a_est
    (a,k), _ = curve_fit(lambda n, a, k : k*n**a,
                         xdata = n,
                         ydata = t,
                         p0 = (a_est, k_est))
    out.write("Fitted k:".ljust(24) + "%.4E\n"%k)
    out.write("Fitted a:".ljust(24) + "%.4E\n"%a)
    out.write("Predicted 1e5 Runtime:".ljust(24) + "%.4E s\n"%(k*1e5**a))


    #Write results from test
    out.write("\n\n% --- Timing Data\n")
    out.write("n             t[s]\n")
    for x1, x2 in zip(n, t):
        out.write(("%.5E"%x1).ljust(14))
        out.write("%.10E\n"%x2)
