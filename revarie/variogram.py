import numpy as np
from scipy.spatial.distance import pdist
from numpy_indexed import group_by
import functools
import warnings

from .fvariogram import fvariogram

class Variogram:
    """

    Calculates lag and squared difference values. Various operations can be
    performed with these quantities within this class.

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
        self.x = x
        self.f = f

        self.check_init()
        self.cond_init()

        self.s = f.size

        self.calc_lags()
        self.calc_diffs()

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
        @functools.wraps(f)
        def wrapper(self, bin_type = "auto", bins = 10, var = False):
            if bin_type == "auto":
                if not int(bins) == bins:
                    raise Exception("Number of bins not correctly castable to"
                            " int")
            elif bin_type == "bound":
                if not all(bins[i] <= bins[i+1] for i in range(len(bins)-1)):
                    raise Exception("Bin boundaries must be arranged in"
                                    " ascending order")
            elif bin_type == "lin":
                if bins[0] > bins[1]:
                    raise Exception("Check format of 'bins' parameter, minima"
                        " should be first element.")
                if int(bins[2]) != bins[2]:
                    raise Exception("Number of bins not correctly castable to"
                            " int")
            else:
                raise Exception("Not known bin type, either auto, bound or li"
                    "n.")

            return f(self, bin_type, bins, var)
        return wrapper

    @_c_matheron
    def matheron(self, bin_type = "auto", bins = 10, var = False):
        #should add binning with constant number of values
        """
        *Calculate Matheron variogram for points and field values previously
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
        bins : int, list, array-like
            Description of how binning will be performed in Matheron
            variogram, specific formats given below for each bin type.
                * "auto" : int giving number of bins to use
                * "lin" : list containing three entries. First is minima of
                    bin boundaries, second is maxima of bin bounds and third
                    is the number of bins to use as int
                * "bound" : array-like object specifying boundaries of bins.
                    Number of bins is length of this array - 1.
        var : bool
            Set True for bin-wise variance to be calculated and returned

        Returns
        -------
        centers : numpy.ndarray
            Bin centers used for variogram
        n_bins : numpy.ndarray
            Number of point relations used to calculate each semivariance
        v : numpy.ndarray
            Estimated semivariance values at lags corresponding to bin centers
        v_var (optional) : numpy.ndarray
            Variance associated with squared difference values within a bin
        """
        bins = self.set_bins(bin_type, bins)
        centers = bins[:-1] + np.diff(bins,1)/2

        b_ind = np.digitize(self.lags, bins)
        n_bins = np.bincount(b_ind-1)[:-1]

        gp = group_by(b_ind[np.where(b_ind != bins.size)])
        _, v = gp.mean(self.diffs[np.where(b_ind != bins.size)])#account for lags bigger than bins
        v = v/2 #SEMI-variogram

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
            return np.linspace(bins[0],bins[1],int(bins[2])+1)-db/2
        elif bin_type == "bound":
            return np.asarray(bins)
        elif bin_type == "auto":
            return np.linspace(self.range[0],self.range[1]/2,int(bins)+1)

    def calc_lags(self):
        """
        Upon initialization, calculates distances between all points given in
        domain. Performed before any reductions are applied.
        """
        self.lags = pdist(self.x)

    def calc_diffs(self):
        """
        Upon initialization, calculates squared differences between all field
        values given. Performed before any reductions are applied.
        """
        c_indx = np.mask_indices(self.s, np.triu, k=1)
        self.diffs = (self.f[c_indx[0]] - self.f[c_indx[1]])**2

    def _c_reduce(f):
        #bin order, bin type etc
        @functools.wraps(f)
        def wrapper(self, typ, bnds, inplace):
            if bnds[0] > bnds [1]:
                raise Exception("Lower and upper bounds out of order.")
            if typ == "abs":
                if bnds[0] < self.range[0]:
                    warnings.warn("Lower bound smaller than smallest lag")
                if bnds[1] > self.range[1]:
                    warnings.warn("Upper bound greater than largest lag")
            elif typ == "quant":
                if not 0 < all(bnds) < 1:
                    raise Exception("Quantile bounds must be between 0 and 1")
            else:
                raise Exception("'{f}' not recognized bound type".fomat(f
                    = typ))

            return f(self, typ, bnds, inplace)
        return wrapper

    @_c_reduce
    def reduce(self, typ, bnds, inplace = True):
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
        @functools.wraps(f)
        def wrapper(self, typ, amnt, inplace):
            if typ == "abs":
                if not 0 < amnt < self.lags.size:
                    raise Exception("'amnt' not between 0 and size of"
                                    " original array")
            elif typ == "frac":
                if not 0 < amnt < 1:
                    raise Exception("'amnt' must be between 0 and 1 for"
                                    " fraction reduction type")
            else:
                raise Exception("'{f}' not recognized bound type".format(f
                    = typ))
            return f(self, typ, amnt, inplace)
        return wrapper

    @_c_rreduce
    def rreduce(self, typ, amnt, inplace = True):
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



    def rm_ids(self, ids, inplace = True):
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
        """
        Notify user of errors during initialization of variogram
        """
        if self.x.shape[0] != self.f.shape[0]:
            raise Exception("Number of data points (x) and number of field "
                            "values (f) do not match.")
        if self.f.ndim > 1:
            if self.f.shape[1] == 1:
                self.f = self.f.flatten()
            else:
                raise Exception("Field values data (f) should be 1D")
        if self.x.ndim > 1:
            if self.x.shape[0] < self.x.shape[1]:
                warnings.warn("Dimension of domain exceeds number of data"
                    " points, check that each point is described by row")
        if np.any(np.iscomplex(self.x)):
            raise Exception("Complex numbers not taken in spatial data")
        if np.any(np.iscomplex(self.f)):
            raise Exception("Complex numbers not taken in field values")

    def cond_init(self):
        """
        Perform data format manimpulation where user interference not
        necessary.
        """
        self.x = np.asarray(self.x)
        self.f = np.asarray(self.f)
        if self.x.ndim < 2:
            self.x = self.x.reshape(self.x.size,1)

class AnisoVariogram(Variogram):
    def __init__(self, x, f):
        super().__init__(x, f)

        self.check_aniso_init()

    def calc_lags(self):
        """
        Upon initialization, calculates angles between all points given in
        domain. Performed before any reductions are applied.
        """
        ti = np.triu_indices(self.s, 1)
        xcombos = [m[ti] for m in np.meshgrid(self.x[:,0], self.x[:,0])]
        ycombos = [m[ti] for m in np.meshgrid(self.x[:,1], self.x[:,1])]
        xdists = np.subtract(*xcombos)
        ydists = np.subtract(*ycombos)
        self.lags = np.sqrt(xdists**2 + ydists**2)
        self.angles = np.arctan2(ydists, xdists) % np.pi

    def anisotropic(self,
                    bin_type = "auto",
                    bins = 10,
                    azimuths = 8,
                    azimuth_tol = None,
                    bandwidth = "auto"):

        bin_boundaries = self.set_bins(bin_type, bins)
        bcenters = bin_boundaries[:-1] + np.diff(bin_boundaries,1)/2
        acenters, abounds, azimuth_tol = self.set_azimuths(azimuths,
                                                           azimuth_tol)

        bandwidth = self.set_bandwidth(bandwidth, bin_boundaries, azimuth_tol)
        nbins = bcenters.size

        #calculate bin numbers for lag distances
        b_ind = np.digitize(self.lags, bin_boundaries)

        #valids indicate lags that are in binning range
        valids = np.where((b_ind != 0) &  (b_ind < nbins + 1))
        b_ind_valid = b_ind[valids]
        lags_valid = self.lags[valids]
        angles_valid = self.angles[valids]
        diffs_valid = self.diffs[valids]

        #compute bin numbers for the azimuthal bins
        a_ind = np.digitize(angles_valid, abounds)

        #valids indicate angles that lie in an azimuthal bin
        valids = np.where((a_ind % 2 == 1) |  (a_ind == 0))
        b_ind_valid = b_ind_valid[valids]
        a_ind_valid = a_ind[valids]
        lags_valid = self.lags[valids]
        angles_valid = angles_valid[valids]
        diffs_valid = diffs_valid[valids]


        #compute bandwidth and distance of each point from nearest
        #  azimuthal center
        adevs = np.abs(angles_valid - acenters[(a_ind_valid-1)//2]) #angle b/w center
                                                                    # and point

        bs = lags_valid*np.sin(adevs) # some trig to get distance from center

        #valids indicate lags that are within the bandwidth (and are therefore
        #  valid)
        valids = np.where(bs < bandwidth)
        b_ind_valid = b_ind_valid[valids]
        a_ind_valid = a_ind_valid[valids]
        lags_valid = lags_valid[valids]
        angles_valid = angles_valid[valids]
        diffs_valid = diffs_valid[valids]


        # grouping the last bin with the first bin
        a_ind_valid[np.where(a_ind_valid == (abounds.size - 1))] = 1
        acenters = acenters[:-1]

        # combining binning criteria to make one bin parameter
        ad_bins = (a_ind_valid//2)*nbins + b_ind_valid

        gp = group_by(ad_bins)
        bids, v_g = gp.mean(diffs_valid)
        v_g = v_g/2

        #build data struct to hold variogram
        v = np.zeros(bcenters.size*acenters.size)
        v[bids-1] = v_g
        v = v.reshape([-1, nbins])

        n_bins = np.bincount(ad_bins-1).reshape([-1, nbins])


        return bcenters, acenters, n_bins.T, v.T

    def set_azimuths(self, azimuths, azimuth_tol):
        centers = np.linspace(0, np.pi, azimuths + 1)

        bounds = np.zeros(centers.size*2)
        if azimuth_tol:
            pass
        else:
            da = centers[1] - centers[0]
            azimuth_tol = da/2

        bounds[2::2] = centers[1:] - azimuth_tol
        bounds[1:-1:2] = centers[:-1] + azimuth_tol
        bounds[-1] = np.pi

        return centers, bounds, azimuth_tol

    def set_bandwidth(self, bandwidth, bins, azimuth_tol):
        if bandwidth == "auto":
            return bins[1]*np.tan(azimuth_tol)
        else:
            return bandwidth


    def rm_ids(self, ids, inplace = False):
        """
        Helper function for reduction methods.

        inplace True highly reccomended
        """
        if inplace:
            self.lags = self.lags[ids]
            self.diffs = self.diffs[ids]
            self.angles = self.angles[ids]
            self.range = (np.min(self.lags), np.max(self.lags))
            self.reduce = True
        else:
            new = Variogram(self.x, self.f)
            new.lags = new.lags[ids]
            new.diffs = new.diffs[ids]
            new.angles = new.angles[ids]
            new.range = (np.min(new.lags), np.max(new.lags))
            new.reduce = True
            return new

    def check_aniso_init(self):
        """
        Run checks on anisotropic variogram that are not run on the isotropic
        variogram
        """
        if self.x.shape[1] != 2:
            raise Exception("Currently, only anisotropic variogram calculatio"
                "supported for 2D fields")



