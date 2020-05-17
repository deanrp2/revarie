#TODO: make extrapolation errors

import numpy as np
import matplotlib.pyplot as plt
import warnings
from functools import partial
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from .models import *

def fvariogram(source, method, options, *args, **kwargs):
    if source == "func":
        if method == "ufunc":
            return options[0]
        else:

            if method == "sph":
                f = spherical
            elif method == "exp":
                f = exponential
            elif method == "gaus":
                f = gaussian

            _f = lambda *a : f(*a[::-1])
            return partial(_f, *options[::-1])

    elif source == "data":
        h = options[0]
        v = options[1]

        if method == "interp":
            return interp(h, v, options[2], *args, **kwargs)
        elif method == "poly":
            return polyfit(h, v, options[2], *args, **kwargs)
        elif method == "bmodel":
            return bmodel_fit(h, v, options[2], *args, **kwargs)
        elif method == "umodel":
            return umodel_fit(h, v, options[2], *args, **kwargs)

def interp(h, v, kind, *args, **kwargs):
    """
    Highl reccomended that copy=False is used
    """
    return interp1d(h, v, kind, *args, **kwargs)

def polyfit(h, v, order, *args, **kwargs):
    P = np.polyfit(h,v,order, *args, **kwargs)

    def f(h):
        return np.polyval(P, h)

    return f

def bmodel_fit(h, v, model, *args, **kwargs):
    """
    Current avaliable models = sph, exp, gaus
    """
    if model == "sph":
        m = spherical
    elif model == "exp":
        m = exponential
    elif model == "gaus":
        m = gaussian

    return umodel_fit(h, v, m, *args, **kwargs)


def umodel_fit(h, v, f, *args, **kwargs):
    opts, _ = curve_fit(f, h, v, *args, **kwargs)
    _f = lambda *a : f(*a[::-1])
    return partial(_f, *opts[::-1])
