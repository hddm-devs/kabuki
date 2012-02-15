from __future__ import division

import numpy as np
import numpy.lib.recfunctions as rec
from scipy.stats import uniform, norm
from copy import copy
import random
import pandas as pd

from kabuki import debug_here

def _add_noise(params, noise=.1, exclude_params=()):
    """Add individual noise to each parameter.

        :Arguments:
            params : dict
                Parameter names and values

        :Optional:
            noise : float <default=.1>
                Standard deviation of random gaussian
                variable to add to each parameter.
            exclude_params : tuple <default=()>
                Do not add noise to these parameters.

        :Returns:
            params : dict
                parameters with noise added.

    """

    params = copy(params)

    for param, value in params.iteritems():
        if param not in exclude_params:
            params[param] = np.random.normal(loc=value, scale=noise)

    return params

def gen_rand_data(dist, params, samples=50, subjs=1, subj_noise=.1, exclude_params=()):
    """Generate a random dataset using a user-defined random distribution.

    :Arguments:
        dist : kabuki.utils.scipy_stochastic
            Probability distribution to sample from (has to have
            random function defined).
        params : dict
            Parameters to use for data generation. Two options possible:
                * {'param1': value, 'param2': value2}
                * {'condition1': {'param1': value, 'param2': value2},
                   'condition2': {'param1': value3, 'param2': value4}}
            In the second case, the dataset is generated with multiple conditions
            named after the key and will be sampled using the corresponding parameters.

    :Optional:
        samples : int <default: 50>
            How many values to sample for each condition for each subject.
        subjs : int <default: 1>
            How many subjects to generate data from. Individual subject parameters
            will be normal distributed around the provided parameters with variance
            subj_noise if subjs > 1. If only one subject is simulated no noise is added.
        subj_noise : float <default: .1>
            How much to perturb individual subj parameters.
        exclude_params : tuple <default ()>
            Do not add noise to these parameters.

    :Returns:
        data : numpy structured array
            Will contain the columns 'subj_idx', 'condition' and 'data' which contains
            the random samples.

    """
    from itertools import product

    # Check if only dict of params was passed, i.e. no conditions
    if not isinstance(params[params.keys()[0]], dict):
        params = {'none': params}

    idx = list(product(range(subjs), params.keys(), np.float64(range(samples))))
    data = np.array(idx, dtype=[('subj_idx', np.int32), ('condition', 'S20'), ('data', np.float64)])

    for condition, param in params.iteritems():
        for subj_idx in range(subjs):
            if subjs > 1:
                subj_param = _add_noise(param, noise=subj_noise, exclude_params=exclude_params)
            else:
                subj_param = param
            samples_from_dist = dist.rv.random(size=samples, **subj_param)
            idx = (data['subj_idx'] == subj_idx) & (data['condition'] == condition)
            data['data'][idx] = np.array(samples_from_dist, dtype=np.float64)

    return data


