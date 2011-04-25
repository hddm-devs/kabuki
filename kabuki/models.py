#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import numpy.lib.recfunctions as rec
import pymc as pm
from copy import copy

import kabuki

#@hierarchical
class Prototype(kabuki.Hierarchical):
    param_names = None

    def __init__(self, data, **kwargs):
        self.data = data

    def get_root_param(self, param, tag, pos=None):
        return None

    def get_tau_param(self, param, tag):
        return None
    
    def get_subj_param(self, param_name, parent_mean, parent_tau, subj_idx, pos=None):
        return pm.Normal('%s%i'%(param_name, subj_idx), mu=parent_mean, tau=parent_tau)


    
#@kabuki.hierarchical
class Regression(Prototype):
    param_names = ('theta', 'x')
    
    def get_root_param(self, param, tag):
        return pm.Uniform('%s%s'%(param,tag), lower=0, upper=50)

    def get_tau_param(self, param, tag):
        return None
    
    def get_observed(self, name, subj_data, params, idx=None):
        @deterministic(plot=False)
        def modelled_y(x=params['x'], theta=params['theta']):
            """Return y computed from the straight line model, given the
            inferred true inputs and the model paramters."""
            slope, intercept = theta
            return slope*x + intercept

        return (pm.Normal(name, mu=modelled_y, tau=2, value=subj_data['dependent'], observed=True), modelled_y)

#@kabuki.hierarchical
class ANOVA(Prototype):
    param_names = ('base', 'effect')
    
    def get_root_param(self, param, all_params, tag, pos=None):
        if pos is not None:
            # Check if last element
            if pos[0] == pos[1]:
                param_full_name = '%s%s'%(param,tag)
                # Set parameter so that they sum to zero
                args = tuple([all_params[p] for p in all_params if p.startswith(param)])
                return pm.Deterministic(kabuki.utils.neg_sum,
                                        param_full_name,
                                        param_full_name,
                                        parents={'args':args})

        return pm.Uniform('%s%s'%(param,tag), lower=-5, upper=5)

    def get_tau_param(self, param, all_params, tag):
        return pm.Uniform('%s%s'%(param,tag), lower=0, upper=800, plot=False)
    
    def get_subj_param(self, param_name, parent_mean, parent_tau, subj_idx, all_params, tag, pos=None):
        param_full_name = '%s%s%i'%(param_name, tag, subj_idx)
        if pos is not None:
            if pos[0] == pos[1]:
                # Set parameter so that they sum to zero
                print all_params
                args = tuple([all_params[p][subj_idx] for p in all_params if p.startswith(param_name) and all_params[p][subj_idx] is not None])
                return pm.Deterministic(kabuki.utils.neg_sum,
                                        param_full_name,
                                        param_full_name,
                                        parents={'args':args})
                                        
        return pm.Normal(param_full_name, mu=parent_mean, tau=parent_tau, plot=False)

    def get_observed(self, name, subj_data, params, idx=None):
        return pm.Normal(name, value=subj_data['score'], mu=params['base'][idx]+params['effect'][idx], tau=2,
                         observed=True, plot=False)
    
