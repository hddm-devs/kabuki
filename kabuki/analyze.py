from __future__ import division
import numpy as np
import pymc as pm
import re
from matplotlib.pylab import figure
import matplotlib.pyplot as plt
import sys, os
from copy import copy

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

def convert_model_to_dictionary(model):
    """convert_model_to_dictionary(model)
    transform a set or list of nodes to a dictionary
    """
    d = {}
    for node in model:
        d[node.__name__] = node
    return d

def get_group_nodes(nodes, return_list=False):
    """
    get_group_nodes(model)
    get only the group nodes from the model
    """

    if type(nodes) is dict:
        group_nodes = {}
        for name, node in nodes.iteritems():
            if (re.search('[A-Za-z)][0-9]+$',name) == None) and \
               not name.startswith('Metropolis') and \
               not name.startswith('deviance'):
                group_nodes[name] = node
        if return_list:
            return group_nodes.values()
        else:
            return group_nodes
    else:
        root = [z for z in nodes if re.search('[A-Za-z)][0-9]+$',z.__name__) == None]
        return root

def get_subjs_numbers(mc):
    if isinstance(mc, pm.MCMC):
        nodes = mc.stochastics
    else:
        nodes = mc

    s = [re.search('[0-9]+$',z.__name__) for z in nodes]
    return list(set([int(x) for x in s if x != None]))

def get_subj_nodes(model, startswith=None, i_subj=None):
    """get_subj_nodes(model, i_subj=None):
    return the nodes of subj i_subj. if is_subj is None then return all subjects' node
    if i_subj is -1, return root nodes

    """
    if isinstance(model, pm.MCMC):
        nodes = model.stochastics
    else:
        nodes = model

    if startswith is None:
        startswith = ''

    if i_subj==-1:
        return get_group_nodes(nodes)
    else:
        if type(nodes) is dict:
            nodes = nodes.values()

        if i_subj is None:
            subj = [z for z in nodes if re.search(startswith+'[A-Za-z)][0-9]+$',z.__name__) != None]
        else:
            subj = [z for z in nodes if re.search(startswith+'[A-Za-z)]%d$'%i_subj,z.__name__) != None]

        if type(nodes) is dict:
            return convert_model_to_dictionary(subj)
        else:
            return subj

def print_stats(stats):
    print gen_stats(stats)

def gen_stats(stats):
    """
    print the model's stats in a pretty format
    Input:
        stats - the output of MCMC.stats()
    """
    names = sorted(stats.keys())
    len_name = max([len(x) for x in names])
    f_names  = ['mean', 'std', '2.5q', '25q', '50q', '75q', '97.5', 'mc_err']
    len_f_names = 6

    s = 'name'.center(len_name) + '  '
    for name in f_names:
        s += ' ' + name.center(len_f_names)
    s += '\n'

    for name in names:
        i_stats = stats[name]
        if not np.isscalar(i_stats['mean']):
            continue
        s += "%s: %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f\n" % \
        (name.ljust(len_name), i_stats['mean'], i_stats['standard deviation'],
         i_stats['quantiles'][2.5], i_stats['quantiles'][25],\
         i_stats['quantiles'][50], i_stats['quantiles'][75], \
         i_stats['quantiles'][97.5], i_stats['mc error'])

    return s

def print_group_stats(stats):
    print gen_group_stats(stats)

def gen_group_stats(stats):
    """
    print the model's group stats in a pretty format
    Input:
        stats - the output of MCMC.stats()
    """

    g_stats = {}
    keys = [z for z in stats.keys() if re.match('[0-9]',z[-1]) is None]
    keys.sort()
    for key in keys:
        g_stats[key] = stats[key]
    s = gen_stats(g_stats)

    return s

def plot_posterior_nodes(nodes, bins=50):
    """Plot interpolated posterior of a list of nodes.

    :Arguments:
        nodes : list of pymc.Node's
            List of pymc.Node's to plot the posterior of
        bins : int (default=50)
            How many bins to use for computing the histogram.

    """
    from kabuki.utils import interpolate_trace
    figure()
    lb = min([min(node.trace()[:]) for node in nodes])
    ub = max([max(node.trace()[:]) for node in nodes])
    x_data = np.linspace(lb, ub, 300)

    for node in nodes:
        trace = node.trace()[:]
        #hist = interpolate_trace(x_data, trace, range=(trace.min(), trace.max()), bins=bins)
        hist = interpolate_trace(x_data, trace, range=(lb, ub), bins=bins)
        plt.plot(x_data, hist, label=node.__name__, lw=2.)

    leg = plt.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)


def group_plot(model, params_to_plot=(), n_bins=50, save_to=None):

    if isinstance(model, pm.MCMC):
        nodes = model.stochastics
    else:
        nodes = model
    db = model.mc.db

    for (param_name, param) in model.params_dict.iteritems():
        #check if we need to plot this parameter
        if (len(params_to_plot) > 0) and  (param_name not in params_to_plot):
            continue
        if not param.has_subj_nodes:
            continue
        
        for (node_tag, group_node) in param.group_nodes.iteritems():
            
            #get subj nodes
            g_node_trace = model.mc.db.trace(group_node.__name__)[:]
            subj_nodes = param.subj_nodes[node_tag]

            #create figure
            print "plotting %s" % group_node.__name__
            sys.stdout.flush()
            figure()
            
            #get x axis
            lb = min([min(db.trace(x.__name__)[:]) for x in subj_nodes])
            lb = min(lb, min(g_node_trace))
            ub = max([max(db.trace(x.__name__)[:]) for x in subj_nodes])
            ub = max(ub, max(g_node_trace))
            x_data = np.linspace(lb, ub, n_bins)
            
            #group histogram
            g_hist = np.histogram(g_node_trace,bins=n_bins, range=[lb, ub], normed=True)[0]
            plt.plot(x_data, g_hist, '--', label='group')
            
            #subj histogram
            for i in subj_nodes:
                g_hist = np.histogram(db.trace(i.__name__)[:],bins=n_bins, range=[lb, ub], normed=True)[0]
                plt.plot(x_data, g_hist, label=re.search('[0-9]+$',i.__name__).group())
                
            #legend and title
            leg = plt.legend(loc='best', fancybox=True)
            leg.get_frame().set_alpha(0.5)
            name = group_node.__name__
            plt.title(name)
            plt.gcf().canvas.set_window_title(name)

            if save_to is not None:
                plt.savefig(os.path.join(save_to, "group_%s.png" % name))
                plt.savefig(os.path.join(save_to, "group_%s.pdf" % name))

def compare_all_pairwise(model):
    """Perform all pairwise comparisons of dependent parameter
    distributions (as indicated by depends_on).
     :Stats generated:
        * Mean difference
        * 5th and 95th percentile
    """
    from scipy.stats import scoreatpercentile
    from itertools import combinations
    print "Parameters\tMean difference\t5%\t95%"
    # Loop through dependent parameters and generate stats
    for params in model.group_nodes_dep.itervalues():
        # Loop through all pairwise combinations
        for p0,p1 in combinations(params, 2):
            diff = model.group_nodes[p0].trace() - model.group_nodes[p1].trace()
            perc_5 = scoreatpercentile(diff, 5)
            perc_95 = scoreatpercentile(diff, 95)
            print "%s vs %s\t%.3f\t%.3f\t%.3f" %(p0, p1, np.mean(diff), perc_5, perc_95)


def plot_all_pairwise(model):
    """Plot all pairwise posteriors to find correlations."""
    import matplotlib.pyplot as plt
    import scipy as sp
    import scipy.stats
    from itertools import combinations
    #size = int(np.ceil(np.sqrt(len(data_deps))))
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    # Loop through all pairwise combinations
    for i, (p0, p1) in enumerate(combinations(model.group_nodes.values())):
        fig.add_subplot(6,6,i+1)
        plt.plot(p0.trace(), p1.trace(), '.')
        (a_s, b_s, r, tt, stderr) = sp.stats.linregress(p0.trace(), p1.trace())
        reg = sp.polyval((a_s, b_s), (np.min(p0.trace()), np.max(p0.trace())))
        plt.plot((np.min(p0.trace()), np.max(p0.trace())), reg, '-')
        plt.xlabel(p0.__name__)
        plt.ylabel(p1.__name__)

    plt.draw()

def savage_dickey(pos, post_trace, range=(-.3,.3), bins=40, prior_trace=None, prior_y=None):
    """Calculate Savage-Dickey density ratio test, see Wagenmakers et
    al. 2010 at http://dx.doi.org/10.1016/j.cogpsych.2009.12.001

    :Arguments:
        pos : float
            position at which to calculate the savage dickey ratio at (i.e. the spec hypothesis you want to test)
        post_trace : numpy.array
            trace of the posterior distribution

    :Optional:
         prior_trace : numpy.array
             trace of the prior distribution
         prior_y : numpy.array
             prior density pos
         range : (int,int)
             Range over which to interpolate and plot
         bins : int
             Over how many bins to compute the histogram over

    :Note: Supply either prior_trace or prior_y.

    """

    x = np.linspace(range[0], range[1], bins)

    if prior_trace is not None:
        # Prior is provided as a trace -> histogram + interpolate
        prior_pos = interpolate_trace(pos, prior_trace, range=range, bins=bins)

    elif prior_y is not None:
        # Prior is provided as a density for each point -> interpolate to retrieve positional density
        import scipy.interpolate
        prior_pos = prior_y #scipy.interpolate.InterpolatedUnivariateSpline(x, prior_y)(pos)
    else:
        assert ValueError, "Supply either prior_trace or prior_y keyword arguments"

    # Histogram and interpolate posterior trace at SD position
    posterior_pos = interpolate_trace(pos, post_trace, range=range, bins=bins)

    # Calculate Savage-Dickey density ratio at pos
    sav_dick = prior_pos / posterior_pos

    return sav_dick

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

def group_cond_diff(hm, node, cond1, cond2, threshold=0):
    """
    compute the difference between different condition in a group analysis.
    The function compute for each subject the difference between 'node' under
    condition 'cond1' to 'node' under condition 'cond2'.
    by assuming that each of the differences is normal distributed
    we can easily compute the group mean and group variance of the difference.
    Then the difference is compared to 'threshold' to compute the mass of the
    group pdf which is smaller than 'threshold'

    Input:
        hm - hierachical model
        node - name of node to be analyized
        cond1 - name of condition 1
        cond2 - name of condition 2
        threshold - see description

    Output:
        group_mean - group mean of the differnce
        group_var - group variance of the difference
        mass_under_threshold  - the mass of the group pdf which is smaller than threshold
    """
    import scipy as sp

    name = node
    node_dict = hm.params_include[name].subj_nodes
    n_subjs = hm._num_subjs
    subj_diff = [None]*n_subjs
    #loop over subjs
    subj_diff_mean = np.zeros(n_subjs)
    subj_diff_std = np.zeros(n_subjs)
    for i_subj in range(n_subjs):
        #compute diffrence of traces
        name1 = node_dict[cond1][i_subj].__name__
        name2 = node_dict[cond2][i_subj].__name__
        trace1 = hm.mc.db.trace(name1)[:]
        trace2 = hm.mc.db.trace(name2)[:]
        diff_trace = trace1 - trace2

        #compute stats
        subj_diff_mean[i_subj] = np.mean(diff_trace)
        subj_diff_std[i_subj]= np.std(diff_trace)

    pooled_var = 1. / sum(1. / (subj_diff_std**2))
    pooled_mean = sum(subj_diff_mean / (subj_diff_std**2)) * pooled_var

    mass_under = sp.stats.norm.cdf(threshold,pooled_mean, np.sqrt(pooled_var))

    return pooled_mean, pooled_var, mass_under

def get_traces(model):
    """Returns recarray of all traces in the model.

    :Arguments:
        model : kabuki.Hierarchical submodel or pymc.MCMC model

    :Returns:
        trace_array : recarray

    """
    if isinstance(model, pm.MCMC):
        m = model
    else:
        m = model.mc

    nodes = list(m.stochastics)

    names = [node.__name__ for node in nodes]
    dtype = [(name, np.float) for name in names]
    traces = np.empty(nodes[0].trace().shape[0], dtype=dtype)

    # Store traces in one array
    for name, node in zip(names, nodes):
        traces[name] = node.trace()[:]

    return traces

def logp_trace(model):
    """
    return a trace of logp for model
    """

    #init
    db = model.mc.db
    n_samples = db.trace('deviance').length()
    logp = np.empty(n_samples, np.double)

    #loop over all samples
    for i_sample in xrange(n_samples):
        #set the value of all stochastic to their 'i_sample' value
        for stochastic in model.mc.stochastics:
            try:
                value = db.trace(stochastic.__name__)[i_sample]
                stochastic.value = value

            except KeyError:
                print "No trace available for %s. " % stochastic.__name__

        #get logp
        logp[i_sample] = model.mc.logp

    return logp

def _evaluate_post_pred(sampled_stats, data_stats, evals=None):
    """Evaluate a summary statistics of sampled sets.

    :Arguments:
        sampled_stats : dict
            Map of summary statistic names to distributions
        data_stats : dict
            Map of summary statistic names to the data distribution

    :Returns:
        pandas.DataFrame containing the eval results as columns.
    """

    import pandas as pd

    from scipy.stats import scoreatpercentile, percentileofscore
    from itertools import product

    if evals is None:
        # Generate some default evals
        evals = OrderedDict()
        evals['in credible interval'] = lambda x, y: (scoreatpercentile(x, 97.5) > y) and (scoreatpercentile(x, 2.5) < y)
        evals['quantile'] = percentileofscore
        evals['SEM'] = lambda x, y: (np.mean(x) - y)**2

    # Evaluate all eval-functions
    results = pd.DataFrame(index=sampled_stats.keys(), columns=evals.keys())
    results.index.names = ['stat']
    for stat_name in sampled_stats.iterkeys():
        for eval_name, func in evals.iteritems():
            value = func(sampled_stats[stat_name], data_stats[stat_name])
            assert np.isscalar(value), "eval function %s is not returning scalar." % eval_name
            results.ix[stat_name][eval_name] = value

    return results


def _post_pred_summary_bottom_node(bottom_node, samples=500, stats=None, plot=False, bins=100, evals=None):
    """Create posterior predictive check for a single bottom node."""
    def _calc_stats(data, stats):
        out = {}
        for name, func in stats.iteritems():
            out[name] = func(data)
        return out

    if stats is None:
        stats = OrderedDict((('mean', np.mean), ('std', np.std)))

    ############################
    # Compute stats over data
    data = bottom_node.value
    data_stats = _calc_stats(data, stats)

    ###############################################
    # Initialize posterior sample stats container
    sampled_stats = {}
    for name in stats.iterkeys():
        sampled_stats[name] = np.empty(samples)

    ##############################
    # Sample and generate stats
    for sample in range(samples):
        _parents_to_random_posterior_sample(bottom_node)
        # Generate data from bottom node
        sampled = bottom_node.random()
        sampled_stat = _calc_stats(sampled, stats)

        # Add it the results container
        for name, value in sampled_stat.iteritems():
            sampled_stats[name][sample] = value

    if plot:
        from pymc.Matplot import gof_plot
        for name, value in sampled_stats.iteritems():
            gof_plot(value, data_stats[name], nbins=bins, name=name, verbose=0)

    result = _evaluate_post_pred(sampled_stats, data_stats, evals=evals)

    return result

def post_pred_check(model, samples=500, bins=100, stats=None, evals=None, plot=False):
    """Run posterior predictive check on a model.

    :Arguments:
        model : kabuki.Hierarchical
            Kabuki model over which to compute the ppc on.

    :Optional:
        samples : int
            How many samples to generate for each node.
        bins : int
            How many bins to use for computing the histogram.
        stats : dict
            User-defined statistics to compute (by default mean and std are computed)
            and evaluate over the samples.
            :Example: {'mean': np.mean, 'median': np.median}
        evals : dict
            User-defined evaluations of the statistics (by default 95 percentile and SEM).
            :Example: {'percentile': scoreatpercentile}
        plot : bool
            Whether to plot the posterior predictive distributions.

    :Returns:
        Hierarchical pandas.DataFrame with the different statistics.
    """
    import pandas as pd
    print "Sampling..."
    results = []

    for name, bottom_node in model.bottom_nodes.iteritems():
        if isinstance(bottom_node, np.ndarray):
            # Group model
            results_subj = []
            subjs = []

            for i_subj, bottom_node_subj in enumerate(bottom_node):
                if bottom_node_subj is None or not hasattr(bottom_node_subj, 'random'):
                    continue # Skip non-existant nodes
                subjs.append(i_subj)
                result_subj = _post_pred_summary_bottom_node(bottom_node_subj, samples=samples, bins=bins, evals=evals, stats=stats, plot=plot)
                results_subj.append(result_subj)

            if len(results_subj) != 0:
                result = pd.concat(results_subj, keys=subjs, names=['subj'])
                results.append(result)
        else:
            # Flat model
            if bottom_node is None or not hasattr(bottom_node, 'random'):
                continue # Skip
            result = _post_pred_summary_bottom_node(bottom_node, samples=samples, bins=bins, evals=evals, stats=stats, plot=plot)
            results.append(result)

    return pd.concat(results, keys=model.bottom_nodes.keys(), names=['node'])

def _parents_to_random_posterior_sample(bottom_node, pos=None):
    """Walks through parents and sets them to pos sample."""
    for i, parent in enumerate(bottom_node.parents.itervalues()):
        if not isinstance(parent, pm.Node): # Skip non-stochastic nodes
            continue

        if pos is None:
            # Set to random posterior position
            pos = np.random.randint(0, len(parent.trace()))

        assert len(parent.trace()) >= pos, "pos larger than posterior sample size"
        parent.value = parent.trace()[pos]


def _post_pred_bottom_node(bottom_node, value_range, samples=10, bins=100, axis=None):
    """Calculate posterior predictive for a certain bottom node.

    :Arguments:
        bottom_node : pymc.stochastic
            Bottom node to compute posterior over.

        value_range : numpy.ndarray
            Range over which to evaluate the likelihood.

    :Optional:
        samples : int (default=10)
            Number of posterior samples to use.

        bins : int (default=100)
            Number of bins to compute histogram over.

        axis : matplotlib.axis (default=None)
            If provided, will plot into axis.
    """

    like = np.empty((samples, len(value_range)), dtype=np.float32)
    for sample in range(samples):
        _parents_to_random_posterior_sample(bottom_node)
        # Generate likelihood for parents parameters
        like[sample,:] = bottom_node.pdf(value_range)

    y = like.mean(axis=0)
    try:
        y_std = like.std(axis=0)
    except FloatingPointError:
        print "WARNING! %s threw FloatingPointError over std computation. Setting to 0 and continuing." % bottom_node.__name__
        y_std = np.zeros_like(y)

    #if len(bottom_node.value) != 0:
        # No data assigned to node
    #    return y, y_std

    #hist, ranges = np.histogram(bottom_node.value, normed=True, bins=bins)

    if axis is not None:
        # Plot pp
        axis.plot(value_range, y, label='post pred', color='b')
        axis.fill_between(value_range, y-y_std, y+y_std, color='b', alpha=.8)

        # Plot data
        if len(bottom_node.value) != 0:
            axis.hist(bottom_node.value, normed=True,
                      range=(value_range[0], value_range[-1]), label='data',
                      bins=100, histtype='step', lw=2.)

        axis.set_ylim(bottom=0) # Likelihood and histogram can only be positive

    return (y, y_std) #, hist, ranges)

def plot_posterior_predictive(model, value_range=None, samples=10, columns=3, bins=100, savefig=False, prefix=None, figsize=(8,6)):
    """Plot the posterior predictive of a kabuki hierarchical model.

    :Arguments:

        model : kabuki.Hierarchical
            The (constructed and sampled) kabuki hierarchical model to
            create the posterior preditive from.

        value_range : numpy.ndarray
            Array to evaluate the likelihood over.

    :Optional:

        samples : int (default=10)
            How many posterior samples to generate the posterior predictive over.

        columns : int (default=3)
            How many columns to use for plotting the subjects.

        bins : int (default=100)
            How many bins to compute the data histogram over.

        savefig : bool (default=False)
            Whether to save the figure to a file.

        prefix : str (default=None)
            Save figure into directory prefix

    :Note:

        This function changes the current value and logp of the nodes.

    """

    if value_range is None:
        # Infer from data by finding the min and max from the nodes
        value_range = np.linspace(model.data)

    for name, bottom_node in model.bottom_nodes.iteritems():
        if isinstance(bottom_node, np.ndarray):
            if not hasattr(bottom_node[0], 'pdf'):
                continue # skip nodes that do not define pdf function
        else:
            if not hasattr(bottom_node, 'pdf'):
                continue # skip nodes that do not define pdf function

        fig = plt.figure(figsize=figsize)

        fig.suptitle(name, fontsize=12)
        fig.subplots_adjust(top=0.9, hspace=.4, wspace=.3)
        if isinstance(bottom_node, np.ndarray):
            # Group model
            for i_subj, bottom_node_subj in enumerate(bottom_node):
                if bottom_node_subj is None:
                    continue # Skip non-existant nodes
                ax = fig.add_subplot(np.ceil(len(bottom_node)/columns), columns, i_subj+1)
                ax.set_title(str(i_subj))
                _post_pred_bottom_node(bottom_node_subj, value_range,
                                       axis=ax,
                                       bins=bins)
        else:
            # Flat model
            _post_pred_bottom_node(bottom_node, value_range,
                                   axis=fig.add_subplot(111),
                                   bins=bins)
            plt.legend()

        if savefig:
            if prefix is not None:
                fig.savefig(os.path.join(prefix, name) + '.svg', format='svg')
                fig.savefig(os.path.join(prefix, name) + '.png', format='png')
            else:
                fig.savefig(name + '.svg', format='svg')
                fig.savefig(name + '.png', format='png')


def _check_bottom_node(bottom_node):
    if bottom_node is None:
        print "Bottom node is None!!"
        return False

    # Check if node has name
    if not hasattr(bottom_node, '__name__'):
        print "bottom node has missing __name__ attribute."

    name = bottom_node.__name__

    # Check if parents exist
    if not hasattr(bottom_node, 'parents'):
        print "bottom node %s is missing parents attribute." % name

    # Check if node has value
    if not hasattr(bottom_node, 'value'):
        print "bottom node %s is missing value." % name

    # Check if data is empty
    if len(bottom_node.value) == 0:
        print "values of bottom node %s is emtpy!" % name

    for i, parent in enumerate(bottom_node.parents.itervalues()):
        if not hasattr(parent, '_logp'): # Skip non-stochastic nodes
            continue


def check_model(model):
    for name, bottom_node in model.bottom_nodes.iteritems():
        if isinstance(bottom_node, np.ndarray):
            # Group model
            for i_subj, bottom_node_subj in enumerate(bottom_node):
                _check_bottom_node(bottom_node_subj)
        else:
            # Flat mode
            _check_bottom_node(bottom_node)

