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
    out.write("% --- CPU info\n")
    out.write(get_processor_info() + "\n\n")

    out.write("% --- Test Info\n")
    out.write("Test Date/Time:".ljust(24) + str(datetime.now()) + "\n")
    out.write("Test Type:".ljust(24) + typ + "\n")
    out.write("Nrange:".ljust(24) + str(n.min()) + "-" + str(n.max()))

    out.write("\n\n% --- Timing Data\n")
    #for x1, x2 in zip(


if __name__ == "__main__":
    n = np.array([1,2,3,4,5,6,7,8,9,10])
    t = np.exp(n)
    write(n, t, "t.dat", "variogram")
