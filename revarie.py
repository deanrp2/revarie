import numpy as np
import matplotlib.pyplot as plt
from  scipy.spatial.distance import pdist

from .models import *
from .variogram import *
from .fvariogram import *

class Revarie:
    def __init__(self, x, mu, sill, model):
        self.x = x
        self.mu = mu
        self.sill = sill
        self.model = model

        self.s = x.shape[0]

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






