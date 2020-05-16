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
    def __init__(self, method, d, dform, rang):
        #need to put in some way to check extrpolation
        self.range = rang
        #self.extra = extrapolation
        #method: interp, polyfit, modelfit, umodelfit, gp???
        #dform: matheron, cloud
        #we need self.f and metadata (methd)
        #need range for extrapolation error
        #returned functions need to always check for extrapolation error

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

    def tmodel(self, h, v, model, *args, **kwargs):
        """
        Current avaliable models = sph, exp, gaus
        """
        if model == "sph":
            m = spherical
        elif model == "exp":
            m = exponential
        elif model == "gaus":
            m = gaussian

        args, _ = curve_fit(m, h, v, *args, **kwargs)

        _m = lambda *a : m(*a[::-1])

        return partial(_m, *args[::-1])




    def echeck(self, x):
        if not self.extra:
            if not np.all(np.logical_and(x > self.range[0],
                                         x < self.range[1])):
                raise Exception("Extrapolation detected, change extrapola"
                    "tion parameter to True if this error is undesired")


