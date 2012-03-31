from __future__ import division

import numpy as np
from copy import copy

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
            if np.isscalar(noise):
                params[param] = np.random.normal(loc=value, scale=noise)
            else:
                if noise.has_key(param):
                    params[param] = np.random.normal(loc=value, scale=noise[param])

    return params

def gen_rand_data(dist, params, samples=50, subjs=1, subj_noise=.1, exclude_params=(), column_name='data'):
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
        subj_noise : float or dictionary <default: .1>
            How much to perturb individual subj parameters.
            if float then each parameter will be sampled from a normal distribution with std of subj_noise.
            if dictionary then only parameters that are keys of subj_noise will be sampled, and the std of the sampling
            distribution will be the value associated with them.
        exclude_params : tuple <default ()>
            Do not add noise to these parameters.
        column_name : str <default='data'>
            What to name the data column.

    :Returns:
        data : numpy structured array
            Will contain the columns 'subj_idx', 'condition' and 'data' which contains
            the random samples.
        subj_params : dict mapping condition to list of individual subject parameters
            Tries to be smart and will return direct values if there is only 1 subject
            and no dict if there is only 1 condition.

    """
    from itertools import product

    # Check if only dict of params was passed, i.e. no conditions
    if not isinstance(params[params.keys()[0]], dict):
        params = {'none': params}

    subj_params = {}

    dtype = np.dtype(dist.dtype)

    idx = list(product(range(subjs), params.keys(), range(samples)))
    data = np.array(idx, dtype=[('subj_idx', np.int32), ('condition', 'S20'), (column_name, dtype)])

    for condition, param in params.iteritems():
        subj_params[condition] = []
        for subj_idx in range(subjs):
            if subjs > 1:
                # Sample subject parameters from a normal around the specified parameters
                subj_param = _add_noise(param, noise=subj_noise, exclude_params=exclude_params)
            else:
                subj_param = param
            subj_params[condition].append(subj_param)
            samples_from_dist = dist.rv.random(size=samples, **subj_param)
            idx = (data['subj_idx'] == subj_idx) & (data['condition'] == condition)
            data[column_name][idx] = np.array(samples_from_dist, dtype=dtype)

    # Remove list around subj_params if there is only 1 subject
    if subjs == 1:
        for key, val in subj_params.iteritems():
            subj_params[key] = val[0]

    # Remove dict around subj_params if there is only 1 condition
    if len(subj_params) == 1:
        subj_params = subj_params[subj_params.keys()[0]]

    return data, subj_params

