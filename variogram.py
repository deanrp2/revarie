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

        self.reduced = False

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
            Descriptor of the format of data passed into the bin parameter.
            Can be one of:
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
        """
        Calculate bin boundaries for bin parameters. See self.matheron for
        details.
        """
        if bin_type == "lin":
            db = (bins[1] - bins[0])/(bins[2] - 1)
            return np.linspace(bins[0],bins[1],bins[2]+1)-db/2
        elif bin_type == "bound":
            return np.asarray(bins)
        elif bin_type == "auto":
            return np.linspace(self.range[0],self.range[1]/2,bins+1)

    def calc_lags(self):
        """
        Upon initialization, calculates distances between all points given in
        domain. Performed before any reductions are applied.
        """
        return pdist(self.x)

    def calc_diffs(self):
        """
        Upon initialization, calculates squared differences between all field
        values given. Performed before any reductions are applied.
        """
        c_indx = np.mask_indices(self.s, np.triu, k=1)
        diffs = (self.f[c_indx[0]] - self.f[c_indx[1]])**2
        return diffs

    def _c_reduce(f):
        #bin order, bin type etc
        @functools.wraps(f)
        def wrapper(self, typ, bnds, inplace = True):
            if not isinstance(inplace, bool):
                print("ERROR")
            return f(self, typ, bnds, inplace)
        return wrapper

    @_c_reduce
    def reduce(self, typ, bnds, inplace = True):#"abs", "quant"
        """
        *Reduce the lag domain of a given Variogram object. Removes data that
        lay outside the given lag boundaries. Very useful for reducing run
        times and memory requirements when performing further operations with
        variogram after initialization such as matheron. Object points and
        field values are not affected. Resulting Variogram objects are also
        marked with a self.reduced = True flag to indicate that the
        self.lags and self.diffs are not reflective of all combinations of
        self.x and self.f.

        Parameters
        ----------
        typ : str
            Descriptor of reduction criteria that is passed to the bnds
            parameter. Can be one of:
                * "abs" : eliminate data from self.diffs and self.lags that
                    come from lags greater or less than specified values
                * "quant" : eliminate data from self.diffs and self.lags by
                    quantiles.
        bnds : list
            2-element list to describe reduction criteria. Specific formats
            given below for each typ option.
                * "abs" : List in the format of (min, max) for range of lags
                    to be included after reduction.
                * "quant" : range of quantiles in format of (min, max) to keep
                    after reduction.
        inplace : bool
            Whether or not object is manipulated inplace. If False, will return
            variogram object with same x and f values but with the reduced lag
            domain.

        Returns
        -------
        new (optional) : Variogram
            New Variogram instance with same x and f values but reduced lag
            domain.
        """
        if typ  == "abs":
            min_lag = bnds[0]
            max_lag = bnds[1]
        elif trim_type == "quant":
            min_lag = np.quantile(self.lags, bnds[0])
            max_lag = np.quantile(self.lags, bnds[1])

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
        """
        *Uniformly downsample stored lags and field value differences for
        faster calculations and smaller memory size. Object points and field
        values are not affected. Resulting Variogram objects are also marked
        with a self.reduced = True flag to indicate that the self.lags and
        self.diffs are not reflective of all combinations of self.x and self.f.

        Parameters
        ----------
        typ : str
            Descriptor of format of reduction amount passed to amnt. Can be
            one of:
                * "abs" : Size of remaining lag and field data difference
                array specified as an absolute size.
                * "frac" : Size of remaining lag and field data difference
                array specified as a fraction of original size.
        amnt : int, float
            Amount to reduce lag and field difference data vectors. Specific
            formats given below for each typ option.
                * "abs" : int giving the size of the remaining lag and
                    field data difference vectors.
                * "frac" : float between 0 and 1 describing how much of lag
                    and field difference should remain as fraction of original
                    size
        inplace : bool
            Whether or not object is manipulated inplace. If False, will return
            variogram object with same x and f values but with the reduced lag
            domain.

        Returns
        -------
        new (optional) : Variogram
            New Variogram instance with same x and f values but reduced lag
            domain.
        """
        if typ == "frac":
            size = int(amnt * self.lags.size)
        if typ == "abs":
            size = amnt

        ids = np.choice(self.lags.size, size)
        self.rm_ids(ids, inplace)



    def rm_ids(ids, inplace = False):
        """
        Helper function for reduction methods.
        """
        if inplace:
            self.lags = self.lags[ids]
            self.diffs = self.diffs[ids]
            self.range = (np.min(self.lags), np.max(self.lags))
            self.reduce = True
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

