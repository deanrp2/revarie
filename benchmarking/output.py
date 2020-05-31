import numpy as np
from scipy.optimize import curve_fit
import platform, subprocess

def get_processor_info():
    if platform.system() == "Windows":
        r = platform.processor()
    elif platform.system() == "Linux":
        command = "lscpu"
        r = subprocess.check_output(command, shell=True).strip()
    return r.decode("utf-8")


def write(times, n, fname):
    out = open(fname, "w")
    out.write("% --- CPU info\n\n")
    out.write(get_processor_info() + "\n\n")


if __name__ == "__main__":
    print(get_processor_info())
