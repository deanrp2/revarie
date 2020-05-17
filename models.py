import numpy as np

def spherical(h, nug, sill, rang):
    variogram = nug + sill*np.piecewise(h, [h <= rang,  h > rang],[lambda h: 1.5*(h/rang)-0.5*(h/rang)**3,1])
    return variogram

def exponential(h, nug, sill, rang):
    variogram = nug +sill*(1.-np.exp(-3*h/rang))
    return variogram

def gaussian(h, nug, sill, rang):
    variogram =  nug+sill*(1.-np.exp(-(2*h/rang)**2))
    return variogram

mtags = {"sph" : spherical,
        "exp" : exponential,
        "gaus" : gaussian}
