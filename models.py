import numpy as np

def spherical(h, c0, r, b):
    variogram = b + c0*np.piecewise(h, [h <= r,  h > r],[lambda h: 1.5*(h/r)-0.5*(h/r)**3,1])
    return c0 - variogram
      
def exponential(h, c0, r, b):
    return b + c0 * (1. - np.exp(-3*h /r))

def gaussian(h, c0, r, b):
    return b + c0 * (1. - np.exp(-2*h/r))