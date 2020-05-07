import numpy as np

def spherical(h, nug, sill, rang):
    variogram = nug + sill*np.piecewise(h, [h <= rang,  h > rang],[lambda h: 1.5*(h/rang)-0.5*(h/rang)**3,1])
    return sill - variogram

def exponential(h, nug, sill, rang):
    variogram = nug +sill*(1.-np.exp(-3*h/rang))
    return sill - variogram

def gaussian(h, nug, sill, rang):
    variogram =  nug+sill*(1.-np.exp(-(2*h/rang)**2))
    return sill - variogram
