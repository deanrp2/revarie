import unittest
from revarie import Revarie
import numpy as np

class TestRevarie(unittest.TestCase):
    def test_init(self):
        """
        Test if any errors in initialization of Revarie instance
        """
        x = np.random.uniform(-1,1,(10,2))
        m = lambda h : 1

        r = Revarie(x, 0, 1, m)

        self.assertTrue(r.mu == 0)

    def test_trivial(self):
        """
        Make sure field generation in trivial case is correct
        """
        x = np.random.uniform(-1,1, (5,2))
        m = lambda h : 0

        r = Revarie(x, 1, 0, m, 1e-13)

        self.assertTrue(np.allclose(r.genf(), 1))

    def test_shape(self):
        """
        Make sure length of generated field data array is same length of x
        """
        x = np.random.uniform(-1,1, (10,2))
        m = lambda h : 0

        r = Revarie(x, 1, 0, m, 1e-13)
        self.assertTrue(r.genf().size, x.shape[0])

    def test_cov(self):
        """
        Test generation of covariance matrix for trivial case
        """
        x = np.linspace(0,10,10)
        m = lambda h : 1

        r = Revarie(x, 1, 1, m, 1e-13)
        self.assertTrue(np.allclose(r.cov, np.eye(10)))

    def test_sparse_init(self):
        """
        Test initialization of sparse matrix
        """
        x = np.linspace(0,1,4)
        m = lambda h : np.piecewise(h, [h<=1,h>1],[lambda h : h,1])

        r = Revarie(x, 0, 1, m, sparse = True)
        self.assertTrue(r.mu == 0)

    def test_sparse_cov(self):
        x = np.linspace(0,3,8)
        m = lambda h : np.piecewise(h, [h<=1,h>1],[lambda h : h,1])

        r_dens = Revarie(x, 0, 1, m, sparse = False)
        r_spar = Revarie(x, 0, 1, m, sparse = True)

        self.assertTrue(np.allclose(r_dens.cov, r_spar.cov.toarray()))

