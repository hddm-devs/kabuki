from __future__ import division
import sys, os

import numpy as np
from matplotlib.pylab import figure
import matplotlib.pyplot as plt

import pandas as pd
import pymc as pm
import pymc.progressbar as pbar

from utils import interpolate_trace
from itertools import combinations

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict


def plot_posterior_nodes(nodes, bins=50):
    """Plot interpolated posterior of a list of nodes.

    :Arguments:
        nodes : list of pymc.Node's
            List of pymc.Node's to plot the posterior of
        bins : int (default=50)
            How many bins to use for computing the histogram.

    """
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


def group_plot(model, params_to_plot=(), bins=50, samples=5000, save_to=None):
    def find_min_max(subj_block):
        # find global min and max for plotting
        min = np.inf
        max = -np.inf
        for name, subj in subj_block.iterrows():
            trace = subj['node'].trace()
            min = np.min([min, np.min(trace)])
            max = np.max([max, np.max(trace)])
        return min, max

    assert model.is_group_model, "group plot only works for group models."

    # select non-observed subject nodes
    subj_nodes = model.nodes_db[(model.nodes_db['observed'] == False) & (model.nodes_db['subj'] == True)]

    knode_names = subj_nodes.groupby(['knode_name', 'tag'])

    for (knode_name, tag), subj_block in knode_names:
        min, max = find_min_max(subj_block)

        # plot interpolated subject histograms
        #create figure
        print "plotting %s: %s" % (knode_name, tag)
        sys.stdout.flush()

        plt.figure()
        plt.title("%s: %s" % (knode_name, tag))
        x = np.linspace(min, max, 100)

        ############################################
        # plot subjects
        for name, subj_descr in subj_block.iterrows():
            trace = subj_descr['node'].trace()
            height = interpolate_trace(x, trace, range=(min, max), bins=bins)
            plt.plot(x, height, lw=1., label=str(np.int32(subj_descr['subj_idx'])))

        ###########################################
        # plot group distribution
        node = subj_descr['node']
        group_trace = np.empty(samples, dtype=np.float32)
        for sample in xrange(samples):
            # set parents to random value from their trace
            trace_pos = np.random.randint(0, len(node.trace()))
            for parent in node.extended_parents:
                parent.value = parent.trace()[trace_pos]
            group_trace[sample] = node.random()
            # TODO: What to do in case of deterministic (e.g. transform) node
            #except AttributeError:
            #    group_trace[sample] = node.parents.items()[0].random()

        height = interpolate_trace(x, group_trace, range=(min, max), bins=bins)
        plt.plot(x, height, '--', lw=2., label='group')

        ##########################################
        #legend and title
        leg = plt.legend(loc='best', fancybox=True)
        leg.get_frame().set_alpha(0.5)
        plt.gcf().canvas.set_window_title(knode_name)

        if save_to is not None:
            plt.savefig(os.path.join(save_to, "group_%s.png" % knode_name))
            plt.savefig(os.path.join(save_to, "group_%s.pdf" % knode_name))

def compare_all_pairwise(model):
    """Perform all pairwise comparisons of dependent parameter
    distributions (as indicated by depends_on).
     :Stats generated:
        * Mean difference
        * 5th and 95th percentile
    """
    raise NotImplementedError, "compare_all_pairwise is currently not functional."
    from scipy.stats import scoreatpercentile

    print "Parameters\tMean difference\t5%\t95%"

    # Loop through dependent parameters and generate stats
    for param in model.nodes_db.itervalues():
        if len(param.group_nodes) < 2:
            continue
        # Loop through all pairwise combinations
        for p0,p1 in combinations(param.group_nodes.values(), 2):
            diff = p0.trace() - p1.trace()
            perc_5 = scoreatpercentile(diff, 5)
            perc_95 = scoreatpercentile(diff, 95)
            print "%s vs %s\t%.3f\t%.3f\t%.3f" %(p0.__name__, p1.__name__, np.mean(diff), perc_5, perc_95)


def plot_all_pairwise(model):
    """Plot all pairwise posteriors to find correlations."""
    import scipy as sp
    from itertools import combinations
    #size = int(np.ceil(np.sqrt(len(data_deps))))
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    # Loop through all pairwise combinations
    for i, (p0, p1) in enumerate(combinations(model.get_group_nodes(), 2)):
        fig.add_subplot(6,6,i+1)
        plt.plot(p0.trace(), p1.trace(), '.')
        (a_s, b_s, r, tt, stderr) = sp.stats.linregress(p0.trace(), p1.trace())
        reg = sp.polyval((a_s, b_s), (np.min(p0.trace()), np.max(p0.trace())))
        plt.plot((np.min(p0.trace()), np.max(p0.trace())), reg, '-')
        plt.xlabel(p0.__name__)
        plt.ylabel(p1.__name__)

    plt.draw()

def gelman_rubin(models):
    """
    Calculate the gelman_rubin statistic (R_hat) for every stochastic in the model.
    (Gelman at al 2004, 11.4)
    Input:
        models - list of models
    """
    names = models[0].stoch_by_name.keys()
    R_hat_dict = {}
    num_samples = models[0].stoch_by_name.values()[0].trace().shape[0] # samples
    num_chains = len(models)
    for name  in models[0].stoch_by_name.iterkeys():
        # Calculate mean for each chain
        samples = np.empty((num_chains, num_samples))
        for i,model in enumerate(models):
            samples[i,:] = model.stoch_by_name[name].trace()

        R_hat_dict[name] = pm.diagnostics.gelman_rubin(samples)

    return R_hat_dict

R_hat = gelman_rubin

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
    Compute the difference between different conditions in a group analysis.
    For each subject the function computes the difference between 'node' under
    condition 'cond1' to 'node' under condition 'cond2'.
    By assuming that each of the differences is normal distributed
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

    #loop over subjs
    subj_diff_mean = np.zeros(n_subjs)
    subj_diff_std = np.zeros(n_subjs)
    for i_subj in range(n_subjs):
        #compute difference of traces
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

    from scipy.stats import scoreatpercentile, percentileofscore

    if evals is None:
        # Generate some default evals
        evals = OrderedDict()
        evals['observed'] = lambda x, y: y
        evals['credible'] = lambda x, y: (scoreatpercentile(x, 97.5) > y) and (scoreatpercentile(x, 2.5) < y)
        evals['quantile'] = percentileofscore
        evals['SEM'] = lambda x, y: (np.mean(x) - y)**2
        evals['mahalanobis'] = lambda x, y: np.abs(np.mean(x) - y)/np.std(x)
        evals['mean'] = lambda x,y: np.mean(x)
        evals['std'] = lambda x,y: np.std(x)
        for q in [2.5, 25, 50, 75, 97.5]:
            key = str(q) + 'q'
            evals[key] = lambda x, y, q=q: scoreatpercentile(x, q)

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

def post_pred_check(model, samples=500, bins=100, stats=None, evals=None, plot=False, progress_bar=True):
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
        progress_bar: bool
            Display progress bar while sampling.


    :Returns:
        Hierarchical pandas.DataFrame with the different statistics.
    """
    import pandas as pd
    results = {}

    # Progress bar
    if progress_bar:
        n_iter = len(model.observed_nodes) * model.num_subjs
        bar = pbar.ProgressBar(n_iter)
        bar_iter = 0
    else:
        print "Sampling..."

    for name, obs_descr in model.iter_observeds():
        node = obs_descr['node']

        if progress_bar:
            bar_iter += 1
            bar.animate(bar_iter)

        if node is None or not hasattr(node, 'random'):
            continue # Skip

        results[name] = _post_pred_summary_bottom_node(node, samples=samples, bins=bins, evals=evals, stats=stats, plot=plot)
        if progress_bar:
            bar.animate(n_iter)

    return pd.concat(results, names=['node'])

def _parents_to_random_posterior_sample(bottom_node, pos=None):
    """Walks through parents and sets them to pos sample."""
    for i, parent in enumerate(bottom_node.extended_parents):
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


    if axis is not None:
        # Plot pp
        axis.plot(value_range, y, label='post pred', color='b')
        axis.fill_between(value_range, y-y_std, y+y_std, color='b', alpha=.8)

        # Plot data
        if len(bottom_node.value) != 0:
            axis.hist(bottom_node.value, normed=True, color='r',
                      range=(value_range[0], value_range[-1]), label='data',
                      bins=bins, histtype='step', lw=2.)

        axis.set_ylim(bottom=0) # Likelihood and histogram can only be positive

    return (y, y_std) #, hist, ranges)

def plot_posterior_predictive(model, value_range=None, samples=10, columns=3, bins=100, savefig=False, path=None, figsize=(8,6)):
    """Plot the posterior predictive distribution of a kabuki hierarchical model.

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

        path : str (default=None)
            Save figure into directory prefix

    :Note:

        This function changes the current value and logp of the nodes.

    """

    if value_range is None:
        # Infer from data by finding the min and max from the nodes
        raise NotImplementedError, "value_range keyword argument must be supplied."

    observeds = model.get_observeds()


    for tag, nodes in observeds.groupby('tag'):
        fig = plt.figure(figsize=figsize)
        fig.suptitle(tag, fontsize=12)
        fig.subplots_adjust(top=0.9, hspace=.4, wspace=.3)

        for subj_i, (node_name, bottom_node) in enumerate(nodes.iterrows()):
            if not hasattr(bottom_node['node'], 'pdf'):
                continue # skip nodes that do not define pdf function

            ax = fig.add_subplot(np.ceil(len(nodes)/columns), columns, subj_i+1)
            ax.set_title(str(bottom_node['subj_idx']))
            _post_pred_bottom_node(bottom_node['node'], value_range,
                                   axis=ax, bins=bins)

        if savefig:
            if path is not None:
                fig.savefig(os.path.join(path, tag) + '.svg', format='svg')
                fig.savefig(os.path.join(path, tag) + '.png', format='png')
            else:
                fig.savefig(tag + '.svg', format='svg')
                fig.savefig(tag + '.png', format='png')



