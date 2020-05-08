import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from numpy_indexed import group_by
import functools

class Variogram:
    """

    Calculates lag and squared difference values. Various operations can be
    performed with these quantities.

    """
    def __init__(self, x, f):
        """
        Create variogram and calculate lags and squared differences

        Parameters
        ----------
        x : numpy.ndarray
            Array of shape (m,n) where n is number of points in an
            m-dimensional domain
        f : numpy.ndarray
            Array of field values observed at each of the n points.
        """
        self.check_init()

        self.x = x
        self.f = f

        self.cond_init()

        self.s = f.size

        self.lags = self.calc_lags()
        self.diffs = self.calc_diffs()

        self.range = (np.min(self.lags), np.max(self.lags))

        self.reduced = False #included for the reduction method

    def cloud(self):
        """
        *Convenient way to get calculated lags and squared differences from
        a Variogram instance. Named for variogram cloud plot.

        Returns
        -------
        lags : numpy.ndarray
            Distance between all combinations of points fed into variogram.
        diffs : numpy.ndarray
            Squared difference between all combinations of field values fed
            into variogram
        """
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
        """
        *Calculate Matheron variogram for points and field values previousely
        fed into variogram. A few options for specifying binning exist.

        Parameters
        ----------
        bin_type : str
            Descriptor of the format of parameter to be passed into the bin
            parameter. Can be one of:
                * "auto" : select bounds to be (0, self.range[1]/2), bin
                    centers will be calculated accordingly based on user given
                    number of bins.
                * "lin" : bin boundaries will be linearly spaced based on a
                    user given minima, maxima and number of bins. Bin centers
                    will not fall on given maxima and minima.
                * "bound" : bounds will be completely given by the user as a
                    numpy array
        bins : int, array-like
            Description of how binning will be performed in Matheron
            variogram, specific formats given below for each bin type.
                * "auto" : int giving number of bins to use
                * "lin" : array-like object containint three entries. First is
                    minima of bin boundaries, second is maxima of bin bounds
                    and third is the number of bins to use
                * "bound" : array-like object specifying boundaries of bins.
                    Number of bins is length of this array - 1.
        var : bool
            Set True for bin-wise variance to be calculated and returned

        Returns
        -------
        centers : numpy.array
            Bin centers used for variogram
        n_bins : numpy array
            Number of point relations used to calculate each semivariance
        v : numpy array
            Estimated semivariance values at lags corresponding to bin centers
        v_var (optional) : numpy array
            Variance associated with squared difference values within a bin
        """
        bins = self.set_bins(bin_type, bins)
        centers = bins[:-1] + np.diff(bins,1)/2

        b_ind = np.digitize(self.lags, bins)
        n_bins = np.bincount(b_ind-1)[:-1]

        gp = group_by(b_ind[np.where(b_ind != bins.size)])
        _, v = gp.mean(self.diffs[np.where(b_ind != bins.size)])#account for lags bigger than bins

        if var:
            _, v_var = gp.var(self.diffs[np.where(b_ind != bins.size)])#account for lags bigger than bins
            return centers, n_bins, v, v_var
        else:
            return centers, n_bins, v


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

    def _c_reduce(f):
        #bin order, bin type etc
        @functools.wraps(f)
        def wrapper(self, typ, bnds, inplace = False):
            if not isinstance(inplace, bool):
                print("ERROR")
            return f(self, typ, bnds, inplace)
        return wrapper

    @_c_reduce
    def reduce(self, typ, bnds, inplace = False):#"abs", "quant"
        if typ  == "abs":
            min_lag = bnds[0]
            max_lag = bnds[1]
        elif trim_type == "quant":
            min_lag = np.quantile(self.lags, trim[0])
            max_lag = np.quantile(self.lags, trim[1])

        ids = np.where(min_lag <= self.lags <= max_lag)
        self.rm_ids(ids, inplace)


    def _c_rreduce(f):
        #bin order, bin type etc
        @functools.wraps(f)
        def wrapper(self, typ, amnt, inplace = False):
            if not isinstance(inplace, bool):
                print("ERROR")
            return f(self, typ, amnt, inplace)
        return wrapper

    @_c_rreduce
    def rreduce(self, typ, amnt, inplace = False): #"abs","frac"
        if typ == "frac":
            size = int(amnt * self.lags.size)
        if typ == "abs":
            size = amnt

        ids = np.choice(self.lags.size, size)
        self.rm_ids(ids, inplace)



    def rm_ids(ids, inplace = False):
        if inplace:
            self.lags = self.lags[ids]
            self.diffs = self.diffs[ids]
            self.range = (np.min(self.lags), np.max(self.lags))
        else:
            new = Variogram(self.x, self.f)
            new.lags = new.lags[ids]
            new.diffs = new.diffs[ids]
            new.range = (np.min(new.lags), np.max(new.lags))
            new.reduce = True

            return new

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
    x = np.random.uniform(0,np.pi,(10,2))
    f = np.sin(x[:,0])

    test = Variogram(x,f)
    print(test.diffs)
    #centers, n, v, v_var = test.matheron("lin", [0,np.pi,20], True)

    #plt.plot(centers, v, "k.")
    #plt.plot(centers, v + v_var, "b--")
    #plt.plot(centers, v - v_var, "b--")
    #plt.show()

