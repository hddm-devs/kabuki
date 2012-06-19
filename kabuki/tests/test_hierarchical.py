import kabuki
import numpy as np
import unittest
from nose.tools import raises
import pymc as pm
from utils import HNodeSimple, HNodeSimpleVar, normal_like, sample_from_models, create_test_models
import pandas as pd
from pandas import Series, DataFrame

class TestHierarchicalBreakDown(unittest.TestCase):
    """
    simple tests to see if hierarchical merthods do not raise and error
    """

    #@classmethod
    #def setUpClass(self):
    def setUp(self):

        #load models
        self.models, self.params = create_test_models()

        #run models
        sample_from_models(self.models, n_iter=200)

    def runTest(self):
        pass

    def test_map(self):
        for model in self.models:
            if model.is_group_model:
                model.approximate_map()
            else:
                model.map(runs=2)

    def test_dic_info(self):
        for model in self.models:
            model.dic_info()

    def test_print_stats(self):
        for model in self.models:
            model.print_stats()
            model.print_stats(subj=False)
            if model.is_group_model:
                model.print_stats(subj_idx=1)

    @unittest.skip("Not implemented")
    def load_db(self):
        pass

    @unittest.skip("TODO")
    def test_init_from_existing_model(self):
        new_models = create_test_models()
        for (i_m, pre_model) in enumerate(self.models):
            new_models[i_m].init_from_existing_model(pre_model)

    def test_plot_posteriors(self):
        pass

    @unittest.skip("TODO")
    def test_subj_by_subj_map_init(self):
        models = create_test_models()
        for model in models:
            if model.is_group_model:
                model.subj_by_subj_map_init(runs=1)


class TestModelCreation(unittest.TestCase):
    #@classmethod
    def setUp(self):
        self.n_subj = 3
        data = kabuki.generate.gen_rand_data(normal_like, {'A':{'loc':0, 'scale':1}, 'B': {'loc':0, 'scale':1}}, subjs=self.n_subj)[0]
        data = pd.DataFrame(data)
        data['condition2'] = np.random.randint(2, size=len(data))
        self.data = data

    def runTest(self):
        pass

    def test_simple_no_deps(self):
        m = HNodeSimple(self.data)
        n_nodes = 1 + self.n_subj*2 #loc_g + n_subj * (loc_subj + like)
        self.assertEqual(len(m.nodes_db), n_nodes)

    def test_simple_deps(self):
        m = HNodeSimple(self.data, depends_on={'loc': 'condition'})
        n_nodes = 2 * (1 + self.n_subj*2) #n_conds * (loc_g + n_subj * (loc_subj + like))
        self.assertEqual(len(m.nodes_db), n_nodes)

    @raises(AssertionError)
    def test_assertion_on_wrong_param_name(self):
        HNodeSimple(self.data, depends_on={'non_existant': 'condition'})

    def test_assertion_on_wrong_param_name(self):
        # does not catch if correct argument
        HNodeSimple(self.data, depends_on={'non_existant': 'condition', 'loc': 'condition'})

    def test_simplevar_partly_deps(self):
        m = HNodeSimpleVar(self.data, depends_on={'loc': 'condition'})
        n_nodes = 1 + 1 + 2 * (1 + self.n_subj*2) #loc_std + loc_tau + n_conds * (loc_g + n_subj * (loc_subj + like))
        self.assertEqual(len(m.nodes_db), n_nodes)

    def test_simplevar_deps(self):
        m = HNodeSimpleVar(self.data, depends_on={'loc': 'condition', 'loc_std':'condition'})
        #n_nodes = n_conds * (loc_std + loc_tau + loc_g + n_subj * (loc_subj + like))
        n_nodes = 2 * (1 + 1 + 1 + self.n_subj*2)
        self.assertEqual(len(m.nodes_db), n_nodes)

    def test_simplevar_double_deps_A(self):
        m = HNodeSimpleVar(self.data, depends_on={'loc': 'condition', 'loc_std':'condition2'})
        #n_nodes = 2*v_tau + 2*v_std + 2*v_g + 4*n_subj*(v_subj + like))
        n_nodes = 2 + 2 + 2 + 4 * self.n_subj * 2
        self.assertEqual(len(m.nodes_db), n_nodes)

    def test_simplevar_double_deps_B(self):
        m = HNodeSimpleVar(self.data, depends_on={'loc': ['condition', 'condition2'], 'loc_std':'condition2'})
        #n_nodes = 2*loc_tau + 2*loc_std + 4*loc_g + 4*n_subj*(loc_subj + like))
        n_nodes = 2 + 2 + 4 + 4 * self.n_subj * 2
        self.assertEqual(len(m.nodes_db), n_nodes)


class TestEstimation(unittest.TestCase):
    """
    simple tests to see if hierarchical merthods do not raise and error
    """

    def test_map_approx(self):
        subjs = 40
        data, params_true = kabuki.generate.gen_rand_data(normal_like,
                                                          {'A':{'loc':0, 'scale':1}, 'B': {'loc':2, 'scale':1}},
                                                          subj_noise={'loc':.1}, samples=200, subjs=subjs)

        model = HNodeSimple(data, depends_on={'loc': 'condition'})

        model.approximate_map()

        counter = 0;
        for condition, subj_params in params_true.iteritems():
            nodes = model.nodes_db[model.nodes_db['condition'] == condition]
            for idx, params in enumerate(subj_params):
                nodes_subj = nodes[nodes['subj_idx'] == idx]
                for param_name, value in params.iteritems():
                    if param_name != 'loc':
                        continue
                    node = nodes_subj[nodes_subj.knode_name == param_name + '_subj']
                    assert len(node) == 1, "Only one node should have been left after slicing."
                    abs_error = np.abs(node.map[0] - value)
                    self.assertTrue(abs_error < .2)
                    counter += 1

        # looping for each condition (i.e. twice)
        self.assertEqual(counter, subjs*2)