import numpy as np
import matplotlib.pyplot as plt
from  scipy.spatial.distance import pdist

from .models import *

class Revarie:
    def __init__(self, x, mu, model):
        self.mu = mu
        self.nug = nug
        self.sill = sill
        self.rang = rang

        self.s = x.shape[0]

        if model == "sph":
            self.model = spherical
        elif model == "exp":
            self.model = exponential
        elif model == "gauss":
            self.model = gaussian
        self.genf(x, model)

    def genf(self, x, model):
        if x.ndim < 2:
            x = x.reshape(x.size, 1)

        h_cov = np.zeros((self.s, self.s))
        h_cov[np.diag_indices(self.s)] = self.sill - self.nug
        mask_indices = np.mask_indices(self.s, np.triu, k=1)

        lags = pdist(x)

        h_cov[mask_indices] = self.model(lags, self.nug, self.sill, self.rang)

        self.cov = np.maximum(h_cov,h_cov.T)

    def mnorm(self):
        return np.random.multivariate_normal(np.ones(self.s)*self.mu, self.cov)






if __name__ == "__main__":
    n_pts = 2000
    #np.random.seed(2)
    x = np.zeros((n_pts, 2))
    x[:,1] = np.random.uniform(0,1,n_pts)
    x[:,0] = np.random.uniform(0,1,n_pts)

    nug = 0
    sill = 1
    rang = 0.8

    x = np.linspace(0,1,n_pts)
    #x = np.sort(np.random.uniform(0,n_pts))

    test = Revarie(x, 0,nug,sill,rang,"sph")
    field =test.mnorm()

    plt.figure()
    plt.plot(x, field,"k")
    #plt.scatter(x[:,0],x[:,1],c= field , s = 9)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
