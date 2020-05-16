#TODO: make extrapolation errors

import numpy as np
import matplotlib.pyplot as plt
import warnings
from functools import partial
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from .models import *

class FuncVariogram:
    """

    Data structure to store functional form of variogam defined in a variety
    of ways.

    """
    def __init__(self,
                 source,
                 method,
                 options,
                 *args, **kwargs):

        self.source = source
        self.method = method


        if source == "func":
            self.irange = (-np.inf, np.inf)

            if method == "ufunc":
                self.f = options[0]
            else:

                if method == "sph":
                    f = spherical
                elif method == "exp":
                    f = exponential
                elif method == "gaus":
                    f = gaussian

                _f = lambda *a : f(*a[::-1])
                self.f = partial(_f, *options[::-1])

        elif source == "data":
            if len(options) == 4:
                self.irange = options[3]
            else:
                self.irange = (np.min(options[0]), np.max(options[0]))

            h = options[0]
            v = options[1]

            if method == "interp":
                self.f = self.interp(h, v, options[2], *args, **kwargs)
            elif method == "poly":
                self.f = self.polyfit(h, v, options[2], *args, **kwargs)
            elif method == "bmodel":
                self.f = self.bmodel_fit(h, v, options[2], *args, **kwargs)
            elif method == "umodel":
                self.f = self.umodel_fit(h, v, options[2], *args, **kwargs)

    def interp(self, h, v, kind, *args, **kwargs):
        """
        Highl reccomended that copy=False is used
        """
        return interp1d(h, v, kind, *args, **kwargs)

    def polyfit(self, h, v, order, *args, **kwargs):
        P = np.polyfit(h,v,order, *args, **kwargs)

        def f(h):
            return np.polyval(P, h)

        return f

    def bmodel_fit(self, h, v, model, *args, **kwargs):
        """
        Current avaliable models = sph, exp, gaus
        """
        if model == "sph":
            m = spherical
        elif model == "exp":
            m = exponential
        elif model == "gaus":
            m = gaussian

        return self.umodel_fit(h, v, m, *args, **kwargs)


    def umodel_fit(self, h, v, f, *args, **kwargs):
        opts, _ = curve_fit(f, h, v, *args, **kwargs)
        _f = lambda *a : f(*a[::-1])
        return partial(_f, *opts[::-1])

    def echeck(self, x):
        if not self.extra:
            if not np.all(np.logical_and(x > self.range[0],
                                         x < self.range[1])):
                raise Exception("Extrapolation detected, change extrapola"
                    "tion parameter to True if this error is undesired")


