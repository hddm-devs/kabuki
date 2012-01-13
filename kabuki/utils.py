from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
from copy import copy
import kabuki

def interpolate_trace(x, trace, range=(-1,1), bins=100):
    """Interpolate distribution (from samples) at position x.

    :Arguments:
        x <float>: position at which to evalute posterior.
        trace <np.ndarray>: Trace containing samples from posterior.

    :Optional:
        range <tuple=(-1,1): Bounds of histogram (should be fairly
            close around region of interest).
        bins <int=100>: Bins of histogram (should depend on trace length).

    :Returns:
        float: Posterior density at x.
    """

    import scipy.interpolate

    x_histo = np.linspace(range[0], range[1], bins)
    histo = np.histogram(trace, bins=bins, range=range, normed=True)[0]
    interp = scipy.interpolate.InterpolatedUnivariateSpline(x_histo, histo)(x)

    return interp

def save_csv(data, fname, sep=None):
    """Save record array to fname as csv.

    :Arguments:
        data <np.recarray>: Data array to output.
        fname <str>: File name.

    :Optional:
        sep <str=','>: Separator between columns.

    :SeeAlso: load_csv
    """
    if sep is None:
        sep = ','
    with open(fname, 'w') as fd:
        # Write header
        fd.write(sep.join(data.dtype.names))
        fd.write('\n')
        # Write data
        for line in data:
            line_str = [str(i) for i in line]
            fd.write(sep.join(line_str))
            fd.write('\n')

def load_csv(*args, **kwargs):
    """Load record array from csv.

    :Arguments:
        fname <str>: File name.
        See numpy.recfromcsv

    :Optional:
        See numpy.recfromcsv

    :Note:
        Direct wrapper for numpy.recfromcsv().

    :SeeAlso: save_csv, numpy.recfromcsv
    """
    #read data
    return np.recfromcsv(*args, **kwargs)

def parse_config_file(fname, mcmc=False, load=False, param_names=None):
    """Open, parse and execute a kabuki model as specified by the
    configuration file.

    :Arguments:
        fname <str>: File name of config file.

    :Optional:
        mcmc <bool=False>: Run MCMC on model.
        load <bool=False>: Load from database.
    """
    raise NotImplementedError("Obsolete, todo!")

    import os.path
    if not os.path.isfile(fname):
        raise ValueError("%s could not be found."%fname)

    import ConfigParser

    config = ConfigParser.ConfigParser()
    config.read(fname)

    #####################################################
    # Parse config file
    data_fname = config.get('data', 'load')
    if not os.path.exists(data_fname):
        raise IOError, "Data file %s not found."%data_fname

    try:
        save = config.get('data', 'save')
    except ConfigParser.NoOptionError:
        save = False

    data = np.recfromcsv(data_fname)

    try:
        model_type = config.get('model', 'type')
    except ConfigParser.NoOptionError:
        model_type = 'simple'

    try:
        is_subj_model = config.getboolean('model', 'is_subj_model')
    except ConfigParser.NoOptionError:
        is_subj_model = True

    try:
        no_bias = config.getboolean('model', 'no_bias')
    except ConfigParser.NoOptionError:
        no_bias = True

    try:
        debug = config.getboolean('model', 'debug')
    except ConfigParser.NoOptionError:
        debug = False

    try:
        dbname = config.get('mcmc', 'dbname')
    except ConfigParser.NoOptionError:
        dbname = None

    # Get depends
    depends = {}
    if param_names is not None:
        for param_name in param_names:
            try:
                # Multiple depends can be listed (separated by a comma)
                depends[param_name] = config.get('depends', param_name).split(',')
            except ConfigParser.NoOptionError:
                pass

    # MCMC values
    try:
        samples = config.getint('mcmc', 'samples')
    except ConfigParser.NoOptionError:
        samples = 10000
    try:
        burn = config.getint('mcmc', 'burn')
    except ConfigParser.NoOptionError:
        burn = 5000
    try:
        thin = config.getint('mcmc', 'thin')
    except ConfigParser.NoOptionError:
        thin = 3
    try:
        verbose = config.getint('mcmc', 'verbose')
    except ConfigParser.NoOptionError:
        verbose = 0

    try:
        plot_rt_fit = config.getboolean('stats', 'plot_rt_fit')
    except ConfigParser.NoOptionError, ConfigParser.NoSectionError:
        plot_rt_fit = False

    try:
        plot_posteriors = config.getboolean('stats', 'plot_posteriors')
    except ConfigParser.NoOptionError, ConfigParser.NoSectionError:
        plot_posteriors = False


    print "Creating model..."

    if mcmc:
        if not load:
            print "Sampling... (this can take some time)"
            m.mcmc(samples=samples, burn=burn, thin=thin, verbose=verbose, dbname=dbname)
        else:
            m.mcmc_load_from_db(dbname=dbname)

    if save:
        m.save_stats(save)
    else:
        print m.summary()

    if plot_rt_fit:
        m.plot_rt_fit()

    if plot_posteriors:
        m.plot_posteriors

    return m

def posterior_predictive_check(model, data):
    params = copy(model.params_est)
    if model.model_type.startswith('simple'):
        params['sv'] = 0
        params['sz'] = 0
        params['ster'] = 0
    if model.no_bias:
        params['z'] = params['a']/2.

    data_sampled = _gen_rts_params(params)

    # Check
    return pm.discrepancy(data_sampled, data, .5)

def load_traces_from_db(mc, dbname):
    """Load samples from a database created by an earlier model
    """
    # Open database
    db = pm.database.hdf5.load(dbname)

    # Loop through parameters and set traces
    for node in mc.nodes:
        #loop only not-observed
        if node.observed:
            continue
        node.trace = db.trace(node.__name__)

def stochastic_from_scipy_dist(scipy_dist, **kwargs):
    """
    Return a Stochastic subclass made from a particular SciPy distribution.
    """
    import inspect
    import scipy.stats.distributions as sc_dst
    from pymc.ScipyDistributions import separate_shape_args
    from pymc.distributions import new_dist_class, bind_size

    if scipy_dist.__class__.__name__.find('_gen'):
        scipy_dist = scipy_dist(**kwargs)

    name = scipy_dist.__class__.__name__.replace('_gen','').capitalize()

    (args, varargs, varkw, defaults) = inspect.getargspec(scipy_dist._pdf)

    shape_args = args[2:]
    if isinstance(scipy_dist, sc_dst.rv_continuous):
        dtype=float

        def logp(value, **kwds):
            args, zkwds = separate_shape_args(kwds, shape_args)
            if hasattr(scipy_dist, '_logp'):
                return scipy_dist._logp(value, *args)
            else:
                return np.sum(scipy_dist.logpdf(value,*args,**kwds))

        parent_names = shape_args + ['loc', 'scale']
        defaults = [None] * (len(parent_names)-2) + [0., 1.]

    elif isinstance(scipy_dist, sc_dst.rv_discrete):
        dtype=int

        def logp(value, **kwds):
            args, kwds = separate_shape_args(kwds, shape_args)
            if hasattr(scipy_dist, '_logp'):
                return scipy_dist._logp(value, *args)
            else:
                return np.sum(scipy_dist.logpmf(value,*args,**kwds))

        parent_names = shape_args + ['loc']
        defaults = [None] * (len(parent_names)-1) + [0]
    else:
        return None

    parents_default = dict(zip(parent_names, defaults))

    def random(shape=None, **kwds):
        args, kwds = separate_shape_args(kwds, shape_args)

        if shape is None:
            return scipy_dist.rvs(*args, **kwds)
        else:
            return np.reshape(scipy_dist.rvs(*args, **kwds), shape)

    # Build docstring from distribution
    docstr = name[0]+' = '+name + '(name, '+', '.join(parent_names)+', value=None, shape=None, trace=True, rseed=True, doc=None)\n\n'
    docstr += 'Stochastic variable with '+name+' distribution.\nParents are: '+', '.join(parent_names) + '.\n\n'
    docstr += """
Methods:

    random()
        - draws random value
          sets value to return value

    ppf(q)
        - percent point function (inverse of cdf --- percentiles)
          sets value to return value

    isf(q)
        - inverse survival function (inverse of sf)
          sets value to return value

    stats(moments='mv')
        - mean('m',axis=0), variance('v'), skew('s'), and/or kurtosis('k')


Attributes:

    logp
        - sum(log(pdf())) or sum(log(pmf()))

    cdf
        - cumulative distribution function

    sf
        - survival function (1-cdf --- sometimes more accurate)

    entropy
        - (differential) entropy of the RV.


NOTE: If you encounter difficulties with this object, please try the analogous
computation using the rv objects in scipy.stats.distributions directly before
reporting the bug.
    """

    new_class = new_dist_class(dtype, name, parent_names, parents_default, docstr, logp, random, True, None)
    class newer_class(new_class):
        __doc__ = docstr
        rv = scipy_dist
        def __init__(self, *args, **kwds):
            new_class.__init__(self, *args, **kwds)
            self.args, self.kwds = separate_shape_args(self.parents, shape_args)
            self.frozen_rv = self.rv(self.args, self.kwds)
            self._random = bind_size(self._random, self.shape)

        def _cdf(self):
            """
            The cumulative distribution function of self conditional on parents
            evaluated at self's current value
            """
            return self.rv.cdf(self.value, *self.args, **self.kwds)
        cdf = property(_cdf, doc=_cdf.__doc__)

        def _sf(self):
            """
            The survival function of self conditional on parents
            evaluated at self's current value
            """
            return self.rv.sf(self.value, *self.args, **self.kwds)
        sf = property(_sf, doc=_sf.__doc__)

        def ppf(self, q):
            """
            The percentile point function (inverse cdf) of self conditional on parents.
            Self's value will be set to the return value.
            """
            self.value = self.rv.ppf(q, *self.args, **self.kwds)
            return self.value

        def isf(self, q):
            """
            The inverse survival function of self conditional on parents.
            Self's value will be set to the return value.
            """
            self.value = self.rv.isf(q, *self.args, **self.kwds)
            return self.value

        def stats(self, moments='mv'):
            """The first few moments of self's distribution conditional on parents"""
            return self.rv.stats(moments=moments, *self.args, **self.kwds)

        def _entropy(self):
            """The entropy of self's distribution conditional on its parents"""
            return self.rv.entropy(*self.args, **self.kwds)
        entropy = property(_entropy, doc=_entropy.__doc__)

    newer_class.__name__ = new_class.__name__
    return newer_class

def set_proposal_sd(mc, tau=.1):
    for var in mc.variables:
        if var.__name__.endswith('var'):
            # Change proposal SD
            mc.use_step_method(pm.Metropolis, var, proposal_sd = tau)

    return


if __name__ == "__main__":
    import doctest
    doctest.testmod()
