from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

import pymc as pm

import kabuki

def percentile(x):
    """Return the 5% and 95% percentile of an array.

    >>> percentile(np.linspace(0,100,100))
    (5., 95.)
    """
    sorted_x = np.sort(x)
    perc = (sorted_x[int(.05*len(sorted_x)-1)],
            sorted_x[int(.95*len(sorted_x)-1)])
    return perc

def all_pairs(seq):
    l = len(seq)
    for i in range(l):
        for j in range(i+1, l):
            yield seq[i], seq[j]

def neg_sum(args):
    return -np.sum(args)

def difference_prior(delta):
    # See Wagenmakers et al 2010, equation 14
    if type(delta) is int:
        if delta<=0:
            return 1+delta
        else:
            return 1-delta
    else:
        out = copy(delta)
        out[delta <= 0] = 1+delta[delta <= 0]
        out[delta > 0] = 1-delta[delta > 0]
        return out

def generate_effect_data(true_mean, true_e1, true_e2, true_e3, subjs, N):
    tau = 1/(2.**2) # Precision is inverse of variance.
    tau_subj = 1.

    # Generate subj means
    true_subj_mean = pm.rnormal(true_mean, tau_subj, size=subjs)
    true_subj_e1 = pm.rnormal(true_subj_mean+true_e1, 800)
    true_subj_e2 = pm.rnormal(true_subj_mean+true_e2, 800)
    true_subj_e3 = pm.rnormal(true_subj_mean+true_e3, 800)

    # Generate data for each subj
    N=50
    samples=N*3
    data = np.empty((subjs*samples), dtype=[('subj_idx','i4'), ('score','f4'), ('cond', 'S4')])
    for subj in range(subjs):
        slice = data[subj*samples:(subj+1)*samples]
        slice['subj_idx'] = subj
        slice['score'][0:N] = pm.rnormal(true_subj_e1[subj], tau, size=N)
        slice['score'][N:2*N] = pm.rnormal(true_subj_e2[subj], tau, size=N)
        slice['score'][2*N:3*N] = pm.rnormal(true_subj_e3[subj], tau, size=N)
        slice[0:N]['cond'] = 'e1'
        slice[N:2*N]['cond'] = 'e2'
        slice[2*N:3*N]['cond'] = 'e3'

    return data

def interpolate_trace(x, trace, range=(-1,1), bins=100):
    import scipy.interpolate

    x_histo = np.linspace(range[0], range[1], bins)
    histo = np.histogram(trace, bins=bins, range=range, normed=True)[0]
    interp = scipy.interpolate.InterpolatedUnivariateSpline(x_histo, histo)(x)

    return interp

def uniform(x, lower, upper):
    y = np.ones(x.shape, dtype=np.float)/(upper-lower)
    #y[x<lower] = 0.
    #y[x>upper] = 0.

    return y

def savage_dickey(post_trace, range=(-1,1), bins=100, plot=False, title=None, savefig=None, prior_trace=None, prior_y=None, plot_prior=True, label=None):
    # Calculate Savage-Dickey density ratio test, see Wagenmakers et al 2010
    # Estimate density of posterior
    # Calculate normalized histogram (density)
    x = np.linspace(range[0], range[1], bins)
    if prior_trace is not None:
        prior0 = interpolate_trace(0, prior_trace, range=range, bins=bins)
        prior = interpolate_trace(x, prior_trace, range=range, bins=bins)
    elif prior_y is not None:
        import scipy.interpolate
        prior0 = scipy.interpolate.InterpolatedUnivariateSpline(x, prior_y)(0)
        prior = prior_y
    else:
        assert ValueError, "Supply either prior_trace or prior_y keyword arguments"
        
    posterior0 = interpolate_trace(0, post_trace, range=range, bins=bins)
    posterior = interpolate_trace(x, post_trace, range=range, bins=bins)

    # Calculate Savage-Dickey density ratio at x=0
    sav_dick = posterior0 / prior0

    if plot:
        if label is None:
            label='posterior'
        plt.plot(x, posterior, label=label, lw=2.)
        if plot_prior:
            plt.plot(x, prior, label='prior', lw=2.)
        plt.axvline(x=0, lw=1., color='k')
        plt.ylim(ymin=0)
        if title:
            plt.title(title)
        if savefig:
            plt.savefig('plots/'+savefig+'.png')

    return sav_dick #, prior, posterior, prior0, posterior0
    

def call_mcmc((model_class, data, dbname, rnd, kwargs)):
    # Randomize seed
    np.random.seed(int(rnd))

    model = model_class(data, **kwargs)
    model.mcmc(dbname=dbname)
    model.mcmc_model.db.close()

def create_tag_names(tag, chains=None):
    import multiprocessing
    if chains is None:
        chains = multiprocessing.cpu_count()
    tag_names = []
    # Create copies of data and the corresponding db names
    for chain in range(chains):
        tag_names.append("db/mcmc%s%i.pickle"% (tag,chain))

    return tag_names

def load_parallel_chains(model_class, data, tag, kwargs, chains=None, test_convergance=True, combine=True):
    tag_names = create_tag_names(tag, chains=chains)
    models = []
    for tag_name in tag_names:
        model = model_class(data, **kwargs)
        model.mcmc_load_from_db(tag_name)
        models.append(model)

    if test_convergance:
        Rhat = test_chain_convergance(models)
        print Rhat
    
    if combine:
        m = combine_chains(models, model_class, data, kwargs)
        return m
    
    return models

def combine_chains(models, model_class, data, kwargs):
    """Combine multiple model runs into one final model (make sure that chains converged)."""
    # Create model that will contain the other traces
    m = copy(models[0])

    # Loop through models and combine chains
    for model in models[1:]:
        m._set_traces(m.group_params, mcmc_model=model.mcmc_model, add=True)
        m._set_traces(m.group_params_tau, mcmc_model=model.mcmc_model, add=True)
        m._set_traces(m.subj_params, mcmc_model=model.mcmc_model, add=True)

    return m

def run_parallel_chains(model_class, data, tag, load=False, cpus=None, chains=None, **kwargs):
    import multiprocessing
    if cpus is None:
        cpus = multiprocessing.cpu_count()

    tag_names = create_tag_names(tag, chains=chains)
    # Parallel call
    if not load:
        rnds = np.random.rand(len(tag_names))*10000
        pool = multiprocessing.Pool(processes=cpus)
        pool.map(call_mcmc, [(model_class, data, tag_name, rnd, kwargs) for tag_name,rnd in zip(tag_names, rnds)])

    models = load_parallel_chains(model_class, data, tag_names, kwargs)

    return models

def R_hat(samples):
    n, num_chains = samples.shape # n=num_samples
    chain_means = np.mean(samples, axis=1)
    # Calculate between-sequence variance
    between_var = n * np.var(chain_means, ddof=1)

    chain_var = np.var(samples, axis=1, ddof=1)
    within_var = np.mean(chain_var)

    marg_post_var = ((n-1.)/n) * within_var + (1./n) * between_var # 11.2
    R_hat_sqrt = np.sqrt(marg_post_var/within_var)

    return R_hat_sqrt

def test_chain_convergance(models):
    # Calculate R statistic to check for chain convergance (Gelman at al 2004, 11.4)
    params = models[0].group_params
    R_hat_param = {}
    for param_name in params.iterkeys():
        # Calculate mean for each chain
        num_samples = models[0].group_params[param_name].trace().shape[0] # samples
        num_chains = len(models)
        samples = np.empty((num_chains, num_samples))
        for i,model in enumerate(models):
            samples[i,:] = model.group_params[param_name].trace()

        R_hat_param[param_name] = R_hat(samples)

    return R_hat_param

def save_csv(data, fname, sep=None):
    """Save record array to fname as csv.
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
    return np.recfromcsv(*args, **kwargs)

def parse_config_file(fname, mcmc=False, load=False, param_names=None):
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
    
    
def check_geweke(model, assert_=True):
    # Test for convergence using geweke method
    for param in model.group_params.itervalues():
        geweke = np.array(pm.geweke(param))
        if assert_:
            assert (np.any(np.abs(geweke[:,1]) < 2)), 'Chain of %s not properly converged'%param
            return False
        else:
            if np.any(np.abs(geweke[:,1]) > 2):
                print "Chain of %s not properly converged" % param
                return False

    return True

if __name__ == "__main__":
    import doctest
    doctest.testmod()
