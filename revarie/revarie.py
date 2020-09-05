import numpy as np
from  scipy.spatial.distance import pdist
import functools
import warnings

from .variogram import *
from .fvariogram import *

class Revarie:
    def __init__(self, x, mu, sill, model, epsilon = 0.):
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
        epsilon : float
            Perturbation amount to supress numerical instabilities in the
            cholesky decomposition
        """
        self.x = x
        self.mu = mu
        self.sill = sill
        self.model = model
        self.s = x.shape[0]

        self.check_init()

        self.calc_cov(x, model)
        self.calc_cholesky(epsilon)

    def calc_cov(self, x, model):
        """
        Creates the covariance matrix from the variogram model
        """
        if x.ndim < 2:
            x = x.reshape(x.size, 1)

        h_cov = np.zeros((self.s, self.s))
        h_cov[np.diag_indices(self.s)] = self.sill
        mask_indices = np.mask_indices(self.s, np.triu, k=1)

        lags = pdist(x)

        h_cov[mask_indices] = self.sill - self.model(lags)

        self.cov = np.maximum(h_cov,h_cov.T)

    def calc_cholesky(self, epsilon):
        """
        Generates cholesky decomposition from covariance matrix for efficient
        random sampling
        """
        pert = epsilon*np.eye(self.cov.shape[0])
        self.chol = np.linalg.cholesky(self.cov + pert)


    def genf(self, n=1):
        """
        Generates random field values using covariance matrix calculated
        previously.

        Parameters
        ----------
        n : int
            Number of fields to be generated

        Returns
        -------
        fvs : numpy array
            Array of field values of shape (dim, n) where each column holds an
            independently generated field. Each row corresponds to an field
            point coordinate value.
        """
        U = np.random.normal(0,1, (self.s, n))
        return np.ones((self.s,1))*self.mu + self.chol@U

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

