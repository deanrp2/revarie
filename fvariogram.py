#TODO: make extrapolation errors

import numpy as np
import matplotlib.pyplot as plt
import warnings
from functools import partial, wraps
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from .models import * #mtags variable comes from here

def _fvariogram(f):
    @wraps(f)
    def wrapper(source, method, options, *args, **kwargs):
        if source == "func":
            if method == "ufunc":
                if not isinstance(options, list) and callable(options):
                    warnings.warn("User-defined function defined in options"
                        " parameter being reformatted as a single element "
                        "list, it is recommended that the user simply enclose"
                        "user-function in list before passing as argument to "
                        "options parameter.")
                    options = [options]
                if len(options) > 1:
                    raise Exception("User-defined function can only take one "
                        "argument, h")
                try:
                    res = options[0](np.zeros(4))
                except:
                    raise Exception("Lags will be passed as numpy array to u"
                        "ser-defined function")
                if not res is np.ndarray:
                    raise Exception("return type of user-defined function sh"
                        " ould be a numpy array")

            elif method in mtags.keys():
                try:
                    mtags[method](*([0] + options))
                except:
                    raise Exception("options parameter not of proper length "
                        "to fill *args of selected built-in model")
            else:
                n = ",".join(["'{g}'".format(g=h) for h in mtags.keys()])
                raise Exception("Method argument for 'func' source sould be "
                    "either 'ufunc' for user specified function or one of the"
                    " built-in models: " + n)

        elif source == "data":
            pass
        else:
            raise Exception("{s} is not a valid value for source parameter,"
                " only 'func' and 'data' are acceptable".format(s=source))

        return f(source, method, options, *args, **kwargs)
    return wrapper


@_fvariogram
def fvariogram(source, method, options, *args, **kwargs):
    if source == "func":
        if method == "ufunc":
            return options[0]
        else:
            f = mtags[method]
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
    m = mtags[model]
    return umodel_fit(h, v, m, *args, **kwargs)


def umodel_fit(h, v, f, *args, **kwargs):
    opts, _ = curve_fit(f, h, v, *args, **kwargs)
    _f = lambda *a : f(*a[::-1])
    return partial(_f, *opts[::-1])
