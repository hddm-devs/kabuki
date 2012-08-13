import kabuki
import pymc as pm
import numpy as np
import unittest
import scipy.stats
from collections import OrderedDict
from kabuki.generate import gen_rand_data, _add_noise

class TestGenerate(unittest.TestCase):
    def runTest(self):
        pass

    def test_add_noise(self):
        noise = 1
        params = OrderedDict([('loc', 0), ('scale', 1)])
        np.random.seed(31337)
        new_params = _add_noise(params, noise=noise)

        # check if copied
        self.assertFalse(new_params is params)

        # check if noise correctly added
        np.random.seed(31337)
        self.assertTrue(new_params['loc'] == np.random.normal(loc=params['loc'], scale=noise))
        self.assertTrue(new_params['scale'] == np.random.normal(loc=params['scale'], scale=noise))

        # test whether exclude works
        new_params = _add_noise(params, noise=noise, exclude_params=('scale',))
        self.assertTrue(new_params['scale'] == 1)

        # test whether bounds work
        for i in range(10):
            bound_params = _add_noise(params, bounds={'loc': (-1, 1), 'scale': (0, 2)}, noise=3)
            assert (bound_params['loc'] > -1) and (bound_params['loc'] < 1)
            assert (bound_params['scale'] > 0) and (bound_params['scale'] < 2)

        # test whether valid_func works
        check_valid_func = lambda **params: (params['loc'] > -1) and (params['loc'] < 1) and (params['scale'] > 0) and (params['scale'] < 2)
        for i in range(10):
            bound_params = _add_noise(params, check_valid_func=check_valid_func, noise=3)
            assert (bound_params['loc'] > -1) and (bound_params['loc'] < 1)
            assert (bound_params['scale'] > 0) and (bound_params['scale'] < 2)

    def test_single_cond_no_subj(self):
        params = {'mu': 0, 'tau': 1}
        seed = 31337
        data, params_return = gen_rand_data(pm.Normal, params, size=100, seed=seed)
        np.random.seed(seed)
        truth = np.float64(pm.rnormal(size=100, **params))

        np.testing.assert_array_equal(data['data'], truth)
        self.assertEqual(params, params_return)

    def test_single_cond_multi_subjs(self):
        params = OrderedDict([('mu', 0), ('tau', 1)])
        subjs = 3
        size = 100

        # generate test data
        seed = 31337
        data, params_subjs = gen_rand_data(pm.Normal, params, size=size, subjs=subjs, seed=seed)

        # test subj present
        np.testing.assert_array_equal(np.unique(data['subj_idx']), range(subjs))

        # test for correct length
        np.testing.assert_array_equal(len(data), subjs*size)

        # generate truth
        np.random.seed(seed)
        for i in range(subjs):
            new_params = _add_noise(params)
            print "check", new_params
            truth = np.float64(pm.rnormal(size=size, **new_params))
            np.testing.assert_array_equal(data[data['subj_idx'] == i]['data'], truth)
            self.assertEqual(params_subjs[i], new_params)

    def test_single_cond_multi_subjs_exclude(self):
        params = OrderedDict([('mu', 0), ('tau', 1)])
        subjs = 3
        size = 100

        # generate test data
        seed = 31337
        data, params_subjs = gen_rand_data(pm.Normal, params, size=size, subjs=subjs,
                                           exclude_params=('tau',), seed=seed)

        # test subj present
        np.testing.assert_array_equal(np.unique(data['subj_idx']), range(subjs))

        # test for correct length
        np.testing.assert_array_equal(len(data), subjs*size)

        # generate truth
        np.random.seed(seed)
        for i in range(subjs):
            new_params = _add_noise(params, exclude_params=('tau',))
            truth = np.float64(pm.rnormal(size=size, **new_params))
            np.testing.assert_array_equal(data[data['subj_idx'] == i]['data'], truth)
            self.assertEqual(params_subjs[i], new_params)


    def test_mulltiple_cond_no_subj(self):
        size = 100
        params = OrderedDict([('cond1', {'mu': 0, 'tau': 1}), ('cond2', {'mu': 100, 'tau': 10})])

        seed = 31337
        data, subj_params = gen_rand_data(pm.Normal, params, size=size, seed=seed)

        # test whether conditions are present
        np.testing.assert_array_equal(np.unique(data['condition']), ['cond1', 'cond2'])
        self.assertEqual(subj_params.keys(), ['cond1', 'cond2'])

        # test for correct length
        np.testing.assert_array_equal(len(data), 2*size)

        # generate truth
        np.random.seed(31337)
        truth = np.float64(pm.rnormal(size=100, **params['cond1']))
        np.testing.assert_array_equal(data[data['condition'] == 'cond1']['data'], truth)

        truth = np.float64(pm.rnormal(size=100, **params['cond2']))
        np.testing.assert_array_equal(data[data['condition'] == 'cond2']['data'], truth)


    def test_column_name(self):
        params = OrderedDict([('mu', 0), ('tau', 1)])
        subjs = 100
        size = 100

        # generate test data
        np.random.seed(31337)
        data, params_subjs = gen_rand_data(pm.Normal, params, size=size, subjs=subjs, exclude_params=('tau',), column_name='test')

        self.assertIn('test', data.dtype.names)
