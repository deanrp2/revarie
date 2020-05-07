import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from numpy_indexed import group_by
import functools

class Exp_Variogram:
    def __init__(self, x, f):
        """
        bin_type: "lin" (minlag, maxlag, nlag) min and max lag 
        samewidth as others, "bound","auto"
        """
        self.check_init()

        self.x = x
        self.f = f

        self.cond_init()

        self.s = f.size

        self.lags = self.calc_lags()
        self.diffs = self.calc_diffs()

        self.range = (np.min(self.lags), np.max(self.lags))

    def cloud(self):
        return self.lags, self.diffs

    def _c_matheron(f):
        #bin order, bin type etc
        @functools.wraps(f)
        def wrapper(self, bin_type, bins, var):
            if not isinstance(bin_type, str):
                print("ERROR")
            return f(self, bin_type, bins, var)
        return wrapper

    @_c_matheron
    def matheron(self, bin_type = "auto", bins = 10, var = False):
        bins = self.set_bins(bin_type, bins)
        centers = bins[:-1] + np.diff(bins,1)/2

        b_ind = np.digitize(self.lags, bins)
        n_bins = np.bincount(b_ind-1)

        gp = group_by(b_ind[np.where(b_ind != bins.size)])
        _, v = gp.mean(self.diffs[np.where(b_ind != bins.size)])#account for lags bigger than bins

        if var:
            _, v_var = gp.var(self.diffs[np.where(b_ind != bins.size)])#account for lags bigger than bins
            return centers, n_bins[:-1], v, v_var
        else:
            return centers, n_bins[:-1], v



    def set_bins(self, bin_type, bins):
        if bin_type == "lin":
            db = (bins[1] - bins[0])/(bins[2] - 1)
            return np.linspace(bins[0],bins[1],bins[2]+1)-db/2
        elif bin_type == "bound":
            return np.asarray(bins)
        elif bin_type == "auto":
            return np.linspace(0,self.range[1]/2,bins+1)

    def calc_lags(self):
        return pdist(self.x)

    def calc_diffs(self):
        c_indx = np.mask_indices(self.s, np.triu, k=1)
        diffs = (self.f[c_indx[0]] - self.f[c_indx[1]])**2
        return diffs


    def check_init(self):
        #check object for valid creation
        # --- x and f same size
        # --- f is 0D
        # --- check that bins and bintype line up
        # --- check if bincenters make any negative lags
        # --- check order of bounds and cent
        # --- check if x is set up correctly shape (ndim, ndata)
        return 0

    def cond_init(self):
        self.x = np.asarray(self.x)
        self.f = np.asarray(self.f)

        if self.x.ndim < 2:
            self.x = self.x.reshape(x.size,1)
        #other object type changes and such




if __name__ == "__main__":
    x = np.random.uniform(0,np.pi,6000)
    f = np.sin(x)

    test = Exp_Variogram(x,f)
    centers, n, v, v_var = test.matheron("lin", [0,np.pi,20], True)

    plt.plot(centers, v, "k.")
    plt.plot(centers, v + v_var, "b--")
    plt.plot(centers, v - v_var, "b--")
    plt.show()

