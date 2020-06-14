import numpy as np
from  scipy.spatial.distance import pdist
import functools
import warnings

from .variogram import *
from .fvariogram import *

class Revarie:
    def __init__(self, x, mu, sill, model):
        """
        Class to generate random fields based on a variogram given as a
        function in the 'model' parameter, mean and variance for a number of
        points given in the x parameter.

        Parameters
        ----------
        x : numpy.ndarray
            Array of shape (m,n) where n is the number of points in an
            m-dimensional domain. Each row is a point. This does not need to
            be the same points used to calculate the original variogram.
        mu : float
            Spatially-independent mean of field values
        sill : float
            Spatially-independent variance of field values
        model : function
            Callable with takes numpy array of lag distances as argument and
            returns numpy array of variogram values. Should only take a single
            parameter.
        """
        self.x = x
        self.mu = mu
        self.sill = sill
        self.model = model
        self.s = x.shape[0]

        self.check_init()

        self.calc_cov(x, model)

    def calc_cov(self, x, model):
        """
        Creates the covariance matrix for the points with the variogram model
        """
        if x.ndim < 2:
            x = x.reshape(x.size, 1)

        h_cov = np.zeros((self.s, self.s))
        h_cov[np.diag_indices(self.s)] = self.sill
        mask_indices = np.mask_indices(self.s, np.triu, k=1)

        lags = pdist(x)

        h_cov[mask_indices] = self.sill - self.model(lags)

        self.cov = np.maximum(h_cov,h_cov.T)

    def genf(self):
        """
        Generates random field values using covariance matrix calculated
        previously.

        Returns
        -------
        fvs : numpy array
            Array of field values of length x.shape[0] that correspond to each
            point in x
        """

        fvs = np.random.multivariate_normal(np.ones(self.s)*self.mu, self.cov)
        return fvs

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

