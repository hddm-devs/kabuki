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
from kabuki.hierarchical import Parameter

from pymc.NumpyDeterministics import deterministic_from_funcs

def where_wrap(condition, x, y):
    """Wrapper for where function"""
    return np.where(condition, x, y) # because numpy functions often don't allow keyword args

stoch_where = deterministic_from_funcs('where',
                                       where_wrap,
                                       jacobians = {'x': lambda condition, x, y: np.where(condition, 1, 0),
                                                    'y': lambda condition, x, y: np.where(condition, 0, 1)},
                                       jacobian_formats = {'x': 'broadcast_operation', 'y': 'broadcast_operation' })

def q_learn(q_mat, lrate, stim, action, reward):
    #if type(q_mat) is not np.ndarray:
    #    q_mat = q_mat.parents['x']
    #select_action = np.zeros_like(q_mat)
    #select_action[stim,action] = 1

    #return stoch_where(select_action, q_mat, q_mat + lrate*(reward-q_mat))

    q_mat_out=np.copy(q_mat)
    # Calc Q-value
    q_val = q_mat[stim,action]
    q_mat_out[stim,action] = q_val + lrate*(reward-q_val)
    return q_mat_out

def softmax(q_mat, inv_temp, stim, action):
    #sm = pm.exp(pm.exp(inv_temp)*q_mat[stim,action]) / (pm.sum(pm.exp(pm.exp(inv_temp)*q_mat[stim,:])))
    sm = np.exp(inv_temp*q_mat[stim,action]) / (np.sum(np.exp(inv_temp*q_mat[stim,:])))
    return sm

def create_choice_probs(data=None, inv_temp=None, lrate=None):
    # Run model
    size = len(data)
    choice_probs = np.empty(size)
    q_vals = []

    for t in range(size):
        if t==0:
            q_val = .5*np.ones((2,2))
        else:
            q_val = q_vals[-1]

        choice_probs[t] = softmax(q_val, inv_temp, data['stim'][t], data['action'][t])
        q_vals.append(q_learn(q_val, lrate, data['stim'][t], data['action'][t], data['reward'][t]))

    return choice_probs

class QLearn(kabuki.Hierarchical):
    def __init__(self, data, **kwargs):
        super(QLearn, self).__init__(data, **kwargs)

        self.num_stims = len(np.unique(data['stim']))
        self.num_actions = len(np.unique(data['action']))

        self.params = [Parameter('lrate', lower=0, upper=1, init=.1),
                       Parameter('inv_temp', lower=1, upper=15, init=5),
                       Parameter('choice_probs', is_bottom_node=True),
                       Parameter('like', is_bottom_node=True)]

    def get_var_node(self, param):
        if param.name == 'lrate':
            return pm.Uniform(param.full_name, lower=0.1, upper=2.,
                              value=1., plot=self.plot_var)
        elif param.name == 'inv_temp':
            return pm.Uniform(param.full_name, lower=.5, upper=5.,
                              value=3., plot=self.plot_var)

    def get_bottom_node(self, param, params):
        #q_vals = np.empty(len(param.data), dtype=object)
        #softmax_probs = np.empty(len(param.data), dtype=object)
        #choice_probs = np.empty(len(param.data), dtype=object)

        if param.name == 'choice_probs':
            return pm.Deterministic(create_choice_probs,
                                    param.full_name,
                                    param.full_name,
                                    {'data':param.data,
                                     'inv_temp':params['inv_temp'],
                                     'lrate':params['lrate']})

        elif param.name == 'like':
            # Create likelihood
            return pm.Bernoulli(param.full_name,
                                p=params['choice_probs'],
                                value=np.ones_like(np.ones_like(params['choice_probs'].value)),
                                observed=True)

        else:
            print "Not found."

#########
# Generate

import numpy as np
import matplotlib.pyplot as plt

def qlearn_generate(trials=50, lrate=.1, rew_prob=.7, inv_temp=5., plot=False, subjs=2):
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



