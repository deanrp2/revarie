import unittest
from revarie import Revarie
import numpy as np

class TestRevarie(unittest.TestCase):
    def test_init(self):
        x = np.random.uniform(-1,1,(10,2))
        m = lambda h : np.sqrt(h)

        r = Revarie(x, 0, 1, m)

        self.assertTrue(r.mu == 0)

    def test_trivial(self):
        x = np.random.uniform(-1,1, (5,2))
        m = lambda h : 0

        r = Revarie(x, 1, 0, m)

        self.assertTrue(np.allclose(r.genf(), 1))

    def test_shape(self):
        x = np.random.uniform(-1,1, (10,2))
        m = lambda h : 0

        r = Revarie(x, 1, 0, m)
        self.assertTrue(r.genf().size, x.shape[0])

    def test_cov(self):
        x = np.linspace(0,10,10)
        m = lambda h : 1

        r = Revarie(x, 1, 1, m)
        self.assertTrue(np.allclose(r.cov, np.eye(10)))


