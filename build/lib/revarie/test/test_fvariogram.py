import unittest
from revarie.fvariogram import *
import numpy as np
from revarie.models import *

class TestFVariogram(unittest.TestCase):
    def test_poly(self):
        """
        Test accuracy of polyfit
        """
        x = np.linspace(0,10,100)

        for n in range(2,8):
            y =  x**n - 2*x**(n-1) + x**(n-2)

            fit = fvariogram(source = "data",
                             method = "poly",
                             options = [x, y, n])
            self.assertTrue(np.allclose(y, fit(x)))

    def test_lin_interp(self):
        """
        Test accuracy of linear interpolation
        """
        x = np.linspace(0,10,100)
        y = .04*x

        fit = fvariogram(source = "data",
                         method = "interp",
                         options = [x, y, "linear"])
        self.assertTrue(np.allclose(y, fit(x)))

    def test_near_interp(self):
        """
        Test accuracy of nearest interploation
        """
        x = np.linspace(0,10,100)
        def f(x):
            return x**2 + x * np.sin(x)
        y = f(x)

        fit = fvariogram(source = "data",
                         method = "interp",
                         options = [x,y,"nearest"])

        xnew = x[:-1] + np.random.uniform(0,1,99)*np.diff(x)
        xnearest = np.array([x[np.abs(x-a).argmin()] for a in xnew])
        self.assertTrue(np.allclose(f(xnearest), fit(xnew)))

    def test_bmodel(self):
        """
        Test accuracy of built-in model fitting
        """
        x = np.linspace(0,10,100)
        y = spherical(x, .2, 4, 6)

        fit = fvariogram(source = "data",
                         method = "bmodel",
                         options = [x,y,"sph"])

        self.assertTrue(np.allclose(y, fit(x)))

    def test_umodel(self):
        """
        Test accuracy of user-specified model fitting
        """
        def u(x, v, h):
            return v*np.sqrt(h*x)

        x = np.linspace(0,10,100)
        y = u(x, 1.4, 3.2)

        fit = fvariogram(source = "data",
                         method = "umodel",
                         options = [x,y,u])

        self.assertTrue(np.allclose(y, fit(x)))

    def test_ufuncs(self):
        """
        Test accuracy of user-specified function specification
        """
        def u(x):
            return np.sqrt(x)

        x = np.linspace(0,10,100)
        yu = u(x)

        ufunc = fvariogram(source = "func",
                           method = "ufunc",
                          options = [u])
        self.assertTrue(np.allclose(yu, ufunc(x)))

    def test_bfuncs(self):
        """
        Test accuracy of built-in function specification
        """
        x = np.linspace(0,10,100)
        yb = spherical(x,.2, 4, 6)

        bfit = fvariogram(source = "func",
                          method = "sph",
                          options = [.2, 4, 6])
        self.assertTrue(np.allclose(yb, bfit(x)))




