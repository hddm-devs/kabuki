from __future__ import division
import numpy as np
import pymc as pm
import re
import kabuki
from matplotlib.pylab import show, hist, close, figure
import matplotlib.pyplot as plt
import sys

def convert_model_to_dictionary(model):
    """convert_model_to_dictionary(model)
    transform a set or list of nodes to a dictionary
    """
    d = {}
    for node in model:
        d[node.__name__] = node
    return d

def get_group_nodes(nodes):
    """
    get_group_nodes(model)
    get only the group nodes from the model
    """
    
    if type(nodes)==type({}):
        nodes = nodes.values()
    
    root = [z for z in nodes if re.search('[A-Za-z)][0-9]+$',z.__name__) == None]
    
    if type(nodes) == type({}):
        return convert_model_to_dictionary(root)
    else:
        return root    
    
def get_subjs_numbers(mc):    
    if type(model) == type(pm.MCMC([])):
        nodes = model.stochastics
    else:
        nodes = model

    s = [re.search('[0-9]+$',z.__name__) for z in nodes]
    return list(set([int(x) for x in s if x != None]))
    
def get_subj_nodes(model, i_subj=None):
    """get_subj_nodes(model, i_subj=None):
    return the nodes of subj i_subj. if is_subj is None then return all subjects' node
    if i_subj is -1, return root nodes
    """ 
    if type(model) == type(pm.MCMC([])):
        nodes = model.stochastics
    else:
        nodes = model


    if i_subj==-1:
        return get_group_nodes(nodes)
    else: 
        if type(nodes) == type({}):
            nodes = nodes.values()
        
        if i_subj is None:        
            subj = [z for z in nodes if re.search('[A-Za-z)][0-9]+$',z.__name__) != None]
        else:
            s_subj = str(i_subj)
            subj = [z for z in nodes if re.search('[A-Za-z)]%d$'%i_subj,z.__name__) != None]
        
        if type(nodes) == type({}):
            return convert_model_to_dictionary(subj)
        else:
            return subj

def print_stats(stats):
    nodes =sorted(stats.keys());    
    len_name = max([len(x) for x in nodes])
    fields = {}
    f_names  = ['mean', 'std', '2.5q', '25q', '50q', '75q', '97.5', 'mc_err']
    len_f_names = 6

    s = 'name'.center(len_name) + '  '
    for name in f_names:
        s = s + ' ' + name.center(len_f_names)
    print s
    for node in nodes:
        v = stats[node]
        if type(v['mean']) == type(np.array([])):
            continue
        print "%s: %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f" % \
        (node.ljust(len_name), v['mean'], v['standard deviation'], 
         v['quantiles'][2.5], v['quantiles'][25],\
         v['quantiles'][50], v['quantiles'][75], \
         v['quantiles'][97.5], v['mc error'])
        
def print_group_stats(stats):
    g_stats = {}
    keys = [z for z in stats.keys() if re.match('[0-9]',z[-1]) is None]
    keys.sort()
    for key in keys:
        g_stats[key] = stats[key]
    print_stats(g_stats)
    
def group_plot(model, n_bins=50):
    if type(model) == type(pm.MCMC([])):
        nodes = model.stochastics
    else:
        nodes = model
    group_nodes = get_group_nodes(nodes)
    
    for node in group_nodes:
        pattern = ('%s[0-9]+'%node.__name__.replace("(","\(")).replace(')','\)')
        subj_nodes = [z for z in nodes if re.search(pattern,z.__name__) != None]
        if subj_nodes == []:
            continue
        
        print "plotting %s" % node.__name__
        sys.stdout.flush()        
        figure()
        subj_nodes = sorted(subj_nodes, key=lambda x:x.__name__)
        lb = min([min(x.trace()) for x in subj_nodes])
        lb = min(lb, min(node.trace()))
        ub = max([max(x.trace()) for x in subj_nodes])
        ub = max(ub, max(node.trace()))
        x_data = np.linspace(lb, ub, n_bins)
        g_hist =np.histogram(node.trace(),bins=n_bins, range=[lb, ub], normed=True)[0]
        plt.plot(x_data, g_hist, '--', label='group')
        for i in subj_nodes:
            g_hist =np.histogram(i.trace(),bins=n_bins, range=[lb, ub], normed=True)[0]
            plt.plot(x_data, g_hist, label=re.search('[0-9]+$',i.__name__).group())
        plt.legend()
        plt.title(node.__name__)
    show()     