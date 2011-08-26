from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm

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
    m = hddm.models.Multi(data, model_type=model_type, is_subj_model=is_subj_model, no_bias=no_bias, depends_on=depends, debug=debug)

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

def set_proposal_sd(mc, tau=.1):
    for var in mc.variables:
        if var.__name__.endswith('var'):
            # Change proposal SD
            mc.use_step_method(pm.Metropolis, var, proposal_sd = tau)

    return

if __name__ == "__main__":
    import doctest
    doctest.testmod()
