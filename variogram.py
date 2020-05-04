import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

class Exp_Variogram:
    def __init__(x, f):
        self.x = self.x
        self.f = self.f
        self.check_init()

        self.s = f.size

        


    def check_init(self):
        #check object for valid creation
        # --- x and f same size
        # --- x and f numpy arrays
        # --- f is 0D
        return 0

if __name__ == "__main__":
    x = np.linspace(0,1,100)
    f = x**2

    test = Exp_Variogram(x,f)
