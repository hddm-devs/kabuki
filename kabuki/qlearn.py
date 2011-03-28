#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import numpy.lib.recfunctions as rec
import pymc as pm
from copy import copy
try:
    from IPython.Debugger import Tracer; debug_here = Tracer()
except:
    def debug_here(): pass


import kabuki

from pymc.NumpyDeterministics import deterministic_from_funcs

def where_wrap(condition, x, y):
    """Wrapper for where function"""
    return np.where(condition, x, y) # because numpy functions often don't allow keyword args

stoch_where = deterministic_from_funcs('where', 
                                       where_wrap, 
                                       jacobians = {'x': lambda condition, x, y: np.where(condition, 1, 0), 
                                                    'y': lambda condition, x, y: np.where(condition, 0, 1)}, 
                                       jacobian_formats = {'x': 'broadcast_operation', 'y': 'broadcast_operation' })

def q_learn(q_mat, lrate_logi, stim, action, reward):
    # lrate is [-inf, inf]. Logistic transform
    lrate = 1. / 1 + pm.exp(-lrate_logi)
    #if type(q_mat) is not np.ndarray:
    #    q_mat = q_mat.parents['x']
    select_action = np.zeros_like(q_mat.value)
    select_action[stim,action] = 1

    return stoch_where(select_action, q_mat, q_mat + lrate*(reward-q_mat))

    q_mat_out=np.copy(q_mat)
    # Calc Q-value
    q_val = q_mat[stim,action]
    q_mat_out[stim,action] = q_val + lrate*(reward-q_val)
    return q_mat_out

def softmax(q_mat, inv_temp, stim, action):
    sm = pm.exp(pm.exp(inv_temp)*q_mat[stim,action]) / (pm.sum(pm.exp(pm.exp(inv_temp)*q_mat[stim,:])))
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
            return pm.Uniform('%s%s'%(param,tag), lower=-6, upper=6)
        elif param == 'inv_temp':
            return pm.Uniform('%s%s'%(param,tag), lower=-6, upper=6)

    def get_tau_param(self, param, all_params, tag):
        return pm.Uniform('%s%s'%(param,tag), lower=0, upper=1.5)
    
    def get_subj_param(self, param_name, parent_mean, parent_tau, subj_idx, all_params, tag, pos=None):
        if param_name == 'lrate':
            return pm.Normal('%s%i'%(param_name, subj_idx), mu=parent_mean, tau=(1/parent_tau**2))
        elif param_name == 'inv_temp':
            return pm.Normal('%s%i'%(param_name, subj_idx), mu=parent_mean, tau=(1/parent_tau**2))

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

            softmax_probs[t] = softmax(q_val, self._reference.group_params['inv_temp'], subj_data['stim'][t], subj_data['action'][t])
            
            choice_probs[t] = pm.Bernoulli('choice_prob_%i_%i'%(idx,t), p=softmax_probs[t], value=1, observed=True)#subj_data['action'][t]


            #q_vals[t] = q_learn(q_val, self._reference.group_params['inv_temp'], subj_data['stim'][t], subj_data['action'][t], subj_data['reward'][t])

            q_vals[t] = q_learn(q_val, 0., subj_data['stim'][t], subj_data['action'][t], subj_data['reward'][t])
                                                  
        return choice_probs









#########
# Generate

import numpy as np
import matplotlib.pyplot as plt

def qlearn_generate(trials=50, lrate=.1, rew_prob=.7, inv_temp=20., plot=False, subjs=2):
    Q = .5*np.ones((2,2))
    out = np.empty(trials*subjs, dtype=[('subj_idx',np.int), ('stim', np.int), ('action', np.int), ('reward', np.float)])
    Q1 = []
    Q2 = []
    for subj in range(subjs):
        idx = slice(subj*trials, (subj+1)*trials)
        out[idx]['subj_idx'] = subj
        for t in range(trials):
            # Present stimulus 0 or 1 (alternating)
            state = t%2
            # Chose action based on softmax over Q-values
            action = softmax_gen(Q[state], inv_temp)

            # Reward is probabilistic (70% for a1 to s1 and vice versa)
            rnd = np.random.rand()
            if action == state:
                reward = 1 if rnd < rew_prob else 0
            else:
                reward = 0 if rnd < rew_prob else 1
            
            # Update Q-value based on reward received
            Q[state,action] = Q[state,action] + lrate * (reward - Q[state,action])
        
            out[idx]['stim'][t] = state
            out[idx]['action'][t] = action
            out[idx]['reward'][t] = reward

            Q1.append(Q[0,0])
            Q2.append(Q[0,1])

    if plot:
        plt.figure()
        plt.plot(Q1)
        plt.plot(Q2)

    return out


def softmax_gen(Q1, inv_temp):
    p1 = np.exp(Q1[0]*inv_temp)
    p1 = p1 / (p1 + np.exp(Q1[1]*inv_temp))

    rand = np.random.rand()
    
    return 0 if rand < p1 else 1



