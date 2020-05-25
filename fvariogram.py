#TODO: make extrapolation errors
#TODO: Check for validity of variogram
#TODO: give user option of returning optimized parameters
#TODO: func->ufunc let user pass parameters

import numpy as np
import matplotlib.pyplot as plt
import warnings
from functools import partial, wraps
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from .models import * #mtags variable comes from here

def _fvariogram(f):
    @wraps(f)
    def wrapper(source, method, options, *args, **kwargs):
        if source == "func":
            if method == "ufunc":#should run model validity tests here as well
                if not isinstance(options, list) and callable(options):
                    warnings.warn("User-defined function defined in options"
                        " parameter being reformatted as a single element "
                        "list, it is recommended that the user simply enclose"
                        "user-function in list before passing as argument to "
                        "options parameter.")
                    options = [options]
                if len(options) > 1:
                    raise Exception("User-defined function can only take one "
                        "argument, h")
                try:
                    res = options[0](np.zeros(4))
                except:
                    raise Exception("Lags will be passed as numpy array to u"
                        "ser-defined function. User-defined functions also m"
                        "ust only take one argument")
                if not res is np.ndarray:
                    raise Exception("return type of user-defined function sh"
                        "ould be a numpy array")

            elif method in mtags.keys():
                try:
                    mtags[method](*([0] + options))
                except:
                    raise Exception("options parameter not of proper length "
                        "to fill *args of selected built-in model")
            else:
                n = ",".join(["'{g}'".format(g=h) for h in mtags.keys()])
                raise Exception("Method argument for 'func' source sould be "
                    "either 'ufunc' for user specified function or one of the"
                    " built-in models: " + n)

        elif source == "data": #left off here
            if method == "poly":
                pass
            elif method == "interp":
                pass
            elif method == "bmodel":
                if options[2] not in mtags.keys():
                    raise Exception("{g} not valid parameter for built-in mod"
                        "el".format(g = options[2]))
            elif method == "umodel":
                pass #this is where we should run validity tests AFTER fitting
        else:
            raise Exception("{s} is not a valid value for source parameter,"
                " only 'func' and 'data' are acceptable".format(s=source))

        return f(source, method, options, *args, **kwargs)
    return wrapper


@_fvariogram
def fvariogram(source, method, options, *args, **kwargs):
    """
    *Function intended to make generating python functions to describe
    variograms centralized. This function can be used to access the built-in
    odels as well as fit around user-defined models. Functionality includes
    fitting variograms to built-in and user-defined models, interpolation and
    polynomial fitting. Put simply, can take a wide range of parameters to
    describe how a function is made that gives variogram as a function of
    distance. Then, returns that function.

    Parameters
    ----------
    source : str
        Descriptor of the general approach used for the variogram model. Can
        be one of:
            * "func" : it is desired to return either a user-defined
                function, or a built-in function with specified parameters.
                This is not the option to select if any sort of fitting is
                desired. See below.
            * "data" : a model will be fit to data that is provided in the
                "options" parameter according to a method given in the
                "method" parameter.
    method : str
        Secondary descriptor which depends on the route selected in the source
        parameter. If "func" was selected, can be one of:
            * "ufunc" : user-specified python function object that takes
                numpy array of lag values and returns variogram values.Should
                only take a single object.
            * built-in model : pass a string of any tag associated with a
                built-in model to use that model with parameters specified
                later in the "options" parameter.

        If "data" is selected, can be one of:
            * "poly" : uses numpy polynomial fit tool to fit a polynomial to
                data provided later in the "options" parameter
            * "interp" : uses the scipy interpolate tools to interpolate data
                provided later in the "options" parameter
            * "bmodel" : fits the data provided in the "options" parameter to
                one of the built-in models using nonlinear least squares
            * "umodel" : fits the data provided in the "options" parameter to
                a user-defined function with an arbitrary number of arguments
                using nonlinear least squares. If values of the arguments are
                known, consider using the source = "func" route because this
                method will fit function parameters to provided data.
    options : list
        List of data needed for the options selected in the previous
        parameters. Descriptions of the content of these lists are given
        below:
            * "func" -> "ufunc" : [<userfunction>]
                <userfunction> user-defined function to be called with
                an array of lags as input that returns an array of variogram
                values.
            * "func" -> built-in model : [*args, **kwargs]
                list of parameters to be passed directly to the selected
                built-in model
            * "data" -> "poly" : [<h>, <v>, <order>, *args, **kwargs]
                <h> array of lag values to fit polynomial to
                <v> array of variogram values to fit polynomial to
                <order> order of polynomial for fitting as int
                *args, **kwargs are passed directly to numpy.polyfit
            * "data" -> "interp" : [<h>, <v>, <kind>, *args, **kwargs]
                <h> array of lag values to use for interpolation
                <v> array of variogram values to use for interpolation
                <kind> str to specify how interpolation is performed, options
                include: "linear", "nearest", "zero", "slinear", "quadratic",
                "cubic", "previous" or "next". See scipy.interpolate.interp1d
                for more details
                *args, **kwargs are passed to scipy.interpolate.interp1d
            * "data" -> "bmodel" : [<h>, <v>, <model tag>, *args, **kwargs]
                <h> array of lag values to fit model to
                <v> array of variogram values to fit model to
                <model tag> string specifying which of the built in models to
                fit
                *args, **kwargs are passed to scipy.optimize.curve_fit
            * "data" -> "umodel" : [<h>, <v>, <user func>, *args, **kwargs]
                <h> array of lag values to fit model to
                <v> array of variogram values to fit model to
                <user func> function object of to use for fitting, first arg
                must be lag values. Following arguments will be found using
                scipy.optimize.curve_fit
                *args, **kwargs are passed to scipy.optimize.curve_fit

            Returns
            -------
            fvariogram : function
                Callable function to be used in later variogram calculations.
                All returned functions will take one argument, lag distance h
                as an array. The return type of all returned functions will be
                an array of returned variogram values at each of those lag
                distances.

    """
    if source == "func":
        if method == "ufunc":
            return options[0]
        else:
            f = mtags[method]
            _f = lambda *a : f(*a[::-1])
            return partial(_f, *options[::-1])

    elif source == "data":
        h = options[0]
        v = options[1]

        if method == "interp":
            return interp(h, v, options[2], *args, **kwargs)
        elif method == "poly":
            return polyfit(h, v, options[2], *args, **kwargs)
        elif method == "bmodel":
            return bmodel_fit(h, v, options[2], *args, **kwargs)
        elif method == "umodel":
            return umodel_fit(h, v, options[2], *args, **kwargs)

def interp(h, v, kind, *args, **kwargs):
    """
    See "data" -> "interp" above
    """
    return interp1d(h, v, kind, *args, **kwargs)

def polyfit(h, v, order, *args, **kwargs):
    """
    See "data" -> "poly" above
    """
    P = np.polyfit(h,v,order, *args, **kwargs)

    def f(h):
        return np.polyval(P, h)

    return f

def bmodel_fit(h, v, model, *args, **kwargs):
    """
    See "data" -> "bmodel" above
    """
    m = mtags[model]
    return umodel_fit(h, v, m, *args, **kwargs)


def umodel_fit(h, v, f, *args, **kwargs):
    """
    See "data" -> "umodel" above
    """
    opts, _ = curve_fit(f, h, v, *args, **kwargs)
    _f = lambda *a : f(*a[::-1])
    return partial(_f, *opts[::-1])
