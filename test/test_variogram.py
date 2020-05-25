import unittest
from revarie import Variogram
import numpy as np
import itertools
from scipy.spatial.distance import pdist


class TestVariogram(unittest.TestCase):
    def test_dist_1D(self):
        """
        Test accuracy of self.lags in 1-D
        """
        x = np.random.uniform(-2,2,8)

        dists = np.zeros(x.size * (x.size - 1) // 2)
        for i, (a,b) in enumerate(itertools.combinations(x,2)):
            dists[i] = np.abs(a-b)

        v = Variogram(x, np.zeros_like(x))

        self.assertTrue(np.allclose(dists, v.lags))

    def test_dist_ND(self):
        """
        Test accuracy of self.lags for N-D
        """
        for n in range(2,5):
            x = np.random.uniform(-2,2,(5,n))
            dists = np.zeros(x.shape[0]*(x.shape[0]-1)//2)

            for i, (a,b) in enumerate(itertools.combinations(x,2)):
                dists[i] = np.sqrt(np.sum(np.square(np.subtract(a,b))))

        v = Variogram(x, np.zeros(x.shape[0]))
        self.assertTrue(np.allclose(dists, v.lags))

    def test_sq_diff(self):
        """
        Test accuracy of self.diffs squared difference
        """
        f = np.random.uniform(-1e3,1e3,20)

        diffs = np.zeros(f.size * (f.size - 1) // 2)

        for i, (a,b) in enumerate(itertools.combinations(f,2)):
            diffs[i] = np.square(np.subtract(a,b))

        v = Variogram(np.zeros_like(f), f)
        self.assertTrue(np.allclose(diffs, v.diffs))

    def test_order(self):
        """
        Test to ensure that lag and squared difference vectors line up in
        order
        """

        d = np.linspace(0,6,12)


        d1 = np.exp(d) #anything monotonic and positive
        v1 = Variogram(d1,d1)
        l1, d1 = v1.cloud()

        self.assertTrue(np.array_equal(np.diff(l1)>0, np.diff(d1)>0))

        d2 = np.sqrt(d)
        v2 = Variogram(d2,d2)
        l2, d2 = v2.cloud()

        self.assertTrue(np.array_equal(np.diff(l2)>0, np.diff(d2)>0))







