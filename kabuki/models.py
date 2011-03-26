#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import numpy.lib.recfunctions as rec
import pymc as pm
from copy import copy

import kabuki

#@hierarchical
class Prototype(object):
    param_names = None

    def __init__(self, data, **kwargs):
        self.data = data

    def get_root_param(self, param, tag):
        return None

    def get_tau_param(self, param, tag):
        return None
    
    def get_subj_param(self, param_name, parent_mean, parent_tau, subj_idx):
        return pm.Normal('%s%i'%(param_name, subj_idx), mu=parent_mean, tau=parent_tau)



def q_learn(q_mat, lrate, stim, action, reward):
    q_mat_out=np.copy(q_mat)
    # Calc Q-value
    q_val = q_mat[stim,action]
    q_mat_out[stim,action] = q_val + lrate*(reward-q_val)
    return q_mat_out

def softmax(q_mat, inv_temp, stim, action):
    sm = np.exp(inv_temp*q_mat[stim,action]) / (np.sum(np.exp(inv_temp*q_mat[stim,:])))
    return sm

@kabuki.hierarchical
class QLearn(object):
    param_names = ('lrate', 'inv_temp')

    def __init__(self, data, **kwargs):
        self.data = data
        self.num_stims = len(np.unique(data['stim']))
        self.num_actions = len(np.unique(data['action']))

    def get_root_param(self, param, all_params, tag):
        if param == 'lrate':
            return pm.Uniform('%s%s'%(param,tag), lower=0, upper=1)
        elif param == 'inv_temp':
            return pm.Uniform('%s%s'%(param,tag), lower=0, upper=10)

    def get_tau_param(self, param, all_params, tag):
        return pm.Uniform('%s%s'%(param,tag), lower=0, upper=100)
    
    def get_subj_param(self, param_name, parent_mean, parent_tau, subj_idx, all_params, tag, pos=None):
        if param_name == 'lrate':
            return pm.TruncatedNormal('%s%i'%(param_name, subj_idx), mu=parent_mean, tau=parent_tau, a=0, b=1)
        elif param_name == 'inv_temp':
            return pm.TruncatedNormal('%s%i'%(param_name, subj_idx), mu=parent_mean, tau=parent_tau, a=0, b=10)

    def get_observed(self, name, subj_data, params, idx=None):
        q_vals = np.empty(len(subj_data), dtype=object)
        softmax_probs = np.empty(len(subj_data), dtype=object)
        choice_probs = np.empty(len(subj_data), dtype=object)

        @pm.deterministic
        def q_val_init():
            return .5*np.ones((self.num_stims, self.num_actions))

        for t in range(len(subj_data)):
            if t==0:
                q_val = q_val_init
            else:
                q_val = q_vals[t-1]

            softmax_probs[t] = pm.Deterministic(softmax, 'softmax_%i_%i'%(idx, t), 'softmax_%i_%i'%(idx, t),
                                                parents={'q_mat': q_val,
                                                         'inv_temp': params['inv_temp'][idx],
                                                         'stim': subj_data['stim'][t],
                                                         'action': subj_data['action'][t]})
            
            choice_probs[t] = pm.Bernoulli('choice_prob_%i_%i'%(idx,t), p=softmax_probs[t], value= 1, observed=True)#subj_data['action'][t]


            q_vals[t] = pm.Deterministic(q_learn, 'q_learn_%i_%i'%(idx, t), 'q_learn_%i_%i'%(idx, t),
                                         parents={'q_mat': q_val,
                                                  'lrate': params['lrate'][idx],
                                                  'stim': subj_data['stim'][t],
                                                  'action': subj_data['action'][t],
                                                  'reward': subj_data['reward'][t]})
                                                  
        return choice_probs

    
@kabuki.hierarchical
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

@kabuki.hierarchical        
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
    
