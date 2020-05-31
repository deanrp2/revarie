import numpy as np
import matplotlib.pyplot as plt
from  scipy.spatial.distance import pdist
import functools
import warnings

from .variogram import *
from .fvariogram import *

class Revarie:
    def __init__(self, x, mu, sill, model):
        self.x = x
        self.mu = mu
        self.sill = sill
        self.model = model
        self.s = x.shape[0]

        self.check_init()

        self.genf(x, model)

    def genf(self, x, model):
        if x.ndim < 2:
            x = x.reshape(x.size, 1)

        h_cov = np.zeros((self.s, self.s))
        h_cov[np.diag_indices(self.s)] = self.sill
        mask_indices = np.mask_indices(self.s, np.triu, k=1)

        lags = pdist(x)

        h_cov[mask_indices] = self.sill - self.model(lags)

        self.cov = np.maximum(h_cov,h_cov.T)

    def mnorm(self):
        return np.random.multivariate_normal(np.ones(self.s)*self.mu, self.cov)

    def check_init(self):
        if not callable(self.model):
            raise Exception("Model initialization parameter must be function"
                    "which takes numpy array of lags as arg and returns corre"
                    "sponding variogram values")
        try:
            self.model(np.zeros(4))
        except:
            raise Exception("Lags will be passed as numpy array to callable d"
                    "efined in model input parameter. Should return numpy arr"
                    "ay as well")

        if self.x.shape[0] < self.x.shape[1]:
            warnings.warn("Dimension of coordinates is larger than number of"
                    " points specified. Each row in 'x' input parameter shoul"
                    "d correspond to a single coordinate.")






