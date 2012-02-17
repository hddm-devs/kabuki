import kabuki
import numpy as np
import unittest
import scipy.stats
from collections import OrderedDict
from kabuki.generate import gen_rand_data, _add_noise

normal_like = kabuki.utils.scipy_stochastic(scipy.stats.distributions.norm_gen, name='normal', longname='normal')

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

    def test_single_cond_no_subj(self):
        params = {'loc': 0, 'scale': 1}
        np.random.seed(31337)
        data, params_return = gen_rand_data(normal_like, params, samples=100)
        np.random.seed(31337)
        truth = np.float64(normal_like.rv.random(size=100, **params))

        np.testing.assert_array_equal(data['data'], truth)
        self.assertEqual(params, params_return)

    def test_single_cond_no_subj_dtype(self):
        params = {'loc': 0, 'scale': 1}
        np.random.seed(31337)
        data, params_return = gen_rand_data(normal_like, params, samples=100, dtype=np.int32)
        np.random.seed(31337)
        truth = np.int32(normal_like.rv.random(size=100, **params))

        np.testing.assert_array_equal(data['data'], truth)
        self.assertEqual(data.dtype[2], np.int32)

        self.assertEqual(params, params_return)

    def test_single_cond_multi_subjs(self):
        params = OrderedDict([('loc', 0), ('scale', 1)])
        subjs = 100
        samples = 100

        # generate test data
        np.random.seed(31337)
        data, params_subjs = gen_rand_data(normal_like, params, samples=samples, subjs=subjs)

        # test subj present
        np.testing.assert_array_equal(np.unique(data['subj_idx']), range(subjs))

        # test for correct length
        np.testing.assert_array_equal(len(data), subjs*samples)

        # generate truth
        np.random.seed(31337)
        for i in range(subjs):
            new_params = _add_noise(params)
            truth = np.float64(normal_like.rv.random(size=samples, **new_params))
            np.testing.assert_array_equal(data[data['subj_idx'] == i]['data'], truth)
            self.assertEqual(params_subjs[i], new_params)

    def test_single_cond_multi_subjs_exclude(self):
        params = OrderedDict([('loc', 0), ('scale', 1)])
        subjs = 100
        samples = 100

        # generate test data
        np.random.seed(31337)
        data, params_subjs = gen_rand_data(normal_like, params, samples=samples, subjs=subjs, exclude_params=('scale',))

        # test subj present
        np.testing.assert_array_equal(np.unique(data['subj_idx']), range(subjs))

        # test for correct length
        np.testing.assert_array_equal(len(data), subjs*samples)

        # generate truth
        np.random.seed(31337)
        for i in range(subjs):
            new_params = _add_noise(params, exclude_params=('scale',))
            truth = np.float64(normal_like.rv.random(size=samples, **new_params))
            np.testing.assert_array_equal(data[data['subj_idx'] == i]['data'], truth)
            self.assertEqual(params_subjs[i], new_params)


    def test_mulltiple_cond_no_subj(self):
        samples = 100
        params = OrderedDict([('cond1', {'loc': 0, 'scale': 1}), ('cond2', {'loc': 100, 'scale': 10})])

        np.random.seed(31337)
        data, subj_params = gen_rand_data(normal_like, params, samples=samples)

        # test whether conditions are present
        np.testing.assert_array_equal(np.unique(data['condition']), ['cond1', 'cond2'])
        self.assertEqual(subj_params.keys(), ['cond1', 'cond2'])

        # test for correct length
        np.testing.assert_array_equal(len(data), 2*samples)

        # generate truth
        np.random.seed(31337)
        truth = np.float64(normal_like.rv.random(size=100, **params['cond1']))
        np.testing.assert_array_equal(data[data['condition'] == 'cond1']['data'], truth)

        truth = np.float64(normal_like.rv.random(size=100, **params['cond2']))
        np.testing.assert_array_equal(data[data['condition'] == 'cond2']['data'], truth)


    def test_column_name(self):
        params = OrderedDict([('loc', 0), ('scale', 1)])
        subjs = 100
        samples = 100

        # generate test data
        np.random.seed(31337)
        data, params_subjs = gen_rand_data(normal_like, params, samples=samples, subjs=subjs, exclude_params=('scale',), column_name='test')

        self.assertIn('test', data.dtype.names)
