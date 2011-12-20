from __future__ import division
import numpy as np
import pymc as pm
import re
import kabuki
from matplotlib.pylab import show, hist, close, figure
import matplotlib.pyplot as plt
import sys
from operator import attrgetter
import scipy as sc

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
    if type(model) is pm.MCMC:
        nodes = model.stochastics
    else:
        nodes = model

    s = [re.search('[0-9]+$',z.__name__) for z in nodes]
    return list(set([int(x) for x in s if x != None]))

def get_subj_nodes(model, startswith=None, i_subj=None):
    """get_subj_nodes(model, i_subj=None):
    return the nodes of subj i_subj. if is_subj is None then return all subjects' node
    if i_subj is -1, return root nodes

    """
    if type(model) == type(pm.MCMC([])):
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
            s_subj = str(i_subj)
            subj = [z for z in nodes if re.search(startswith+'[A-Za-z)]%d$'%i_subj,z.__name__) != None]

        if type(nodes) is dict:
            return convert_model_to_dictionary(subj)
        else:
            return subj

def gen_stats(traces, alpha=0.05, batches=100):
    """Useful helper function to generate stats() on a loaded database
    object.  Pass the db._traces list.

    """

    from pymc.utils import hpd, quantiles
    from pymc import batchsd

    stats = {}
    for name, trace_obj in traces.iteritems():
        trace = np.squeeze(np.array(trace_obj(), float))
        stats[name] = {'standard deviation': trace.std(0),
                       'mean': trace.mean(0),
                       '%s%s HPD interval' % (int(100*(1-alpha)),'%'): hpd(trace, alpha),
                       'mc error': batchsd(trace, batches),
                       'quantiles': quantiles(trace)}

    return stats

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
    fields = {}
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

def group_plot(model, params_to_plot = (), n_bins=50):
    if type(model) is pm.MCMC:
        nodes = model.stochastics
    else:
        nodes = model
    db = model.mc.db

    for (param_name, param) in model.params_dict.iteritems():
        if len(params_to_plot) > 0 and  param_name not in params_to_plot:
            continue
        for (node_tag, group_node) in param.group_nodes.iteritems():
            g_node_trace = model.mc.db.trace(group_node.__name__)[:]
            subj_nodes = param.subj_nodes[node_tag]
            if subj_nodes == []:
                continue

            print "plotting %s" % group_node.__name__
            sys.stdout.flush()
            figure()
            lb = min([min(db.trace(x.__name__)[:]) for x in subj_nodes])
            lb = min(lb, min(g_node_trace))
            ub = max([max(db.trace(x.__name__)[:]) for x in subj_nodes])
            ub = max(ub, max(g_node_trace))
            x_data = np.linspace(lb, ub, n_bins)
            g_hist = np.histogram(g_node_trace,bins=n_bins, range=[lb, ub], normed=True)[0]
            plt.plot(x_data, g_hist, '--', label='group')
            for i in subj_nodes:
                g_hist =np.histogram(db.trace(i.__name__)[:],bins=n_bins, range=[lb, ub], normed=True)[0]
                plt.plot(x_data, g_hist, label=re.search('[0-9]+$',i.__name__).group())
            leg = plt.legend(loc='best', fancybox=True)
            leg.get_frame().set_alpha(0.5)
            plt.title(group_node.__name__)
            plt.gcf().canvas.set_window_title(group_node.__name__)
    show()

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

def group_cond_diff(hm, node, cond1, cond2, threshold = 0):
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

    mass_under = sc.stats.norm.cdf(threshold,pooled_mean, np.sqrt(pooled_var))

    return pooled_mean, pooled_var, mass_under