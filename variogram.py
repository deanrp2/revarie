import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

class Exp_Variogram:
    def __init__(self, x, f, bins, bin_type):
        """
        bin_type: "lin" (minlag, maxlag, nlag) min and max lag 
        samewidth as others, "bound","auto"
        """
        self.check_init()

        self.x = x
        self.f = f
        self.s = f.size

        self.cond_init()

        self.lags = self.calc_lags()
        self.diffs = self.calc_diffs()

        self.set_bins(bins, bin_type)


    def set_bins(self, bins, bin_type):
        if bin_type == "lin":
            db = (bins[1] - bins[0])/(bins[2] - 1)
            self.bin_bounds = np.linspace(bins[0],bins[1],bins[2]+1)-db/2
        elif bin_type == "bound":
            self.bin_bounds = np.asarray(bins)
        elif bin_type == "auto":
            print("auto not yet implimented")

    def calc_lags(self):
        return pdist(self.x)

    def calc_diffs(self):
        c_indx = np.mask_indices(self.s, np.triu, k=1)
        diffs = (self.f[c_indx[0]] - self.f[c_indx[1]])**2
        return diffs

    def check_init(self):
        #check object for valid creation
        # --- x and f same size
        # --- x and f numpy arrays
        # --- f is 0D
        # --- check that bins and bintype line up
        # --- check if bincenters make any negative lags
        # --- check order of bounds and cent
        return 0

    def cond_init(self):
        self.x = np.asarray(self.x)
        self.f = np.asarray(self.f)

        if self.x.ndim < 2:
            self.x = self.x.reshape(x.size,1)
        #other object type changes and such




if __name__ == "__main__":
    x = np.linspace(0,1,10)
    f = x**2

    test = Exp_Variogram(x,f, (1,10,10) ,"lin")
