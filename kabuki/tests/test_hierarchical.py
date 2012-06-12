import kabuki
import numpy as np
import unittest
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
        self.models = create_test_models()

        #run models
        sample_from_models(self.models, n_iter=200)

    def runTest(self):
        pass

    def test_map(self):
        for model in self.models:
            if model.is_group_model:
                model.approx_map()
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
        n_nodes = 1 + self.n_subj*2 #v_g + n_subj * (v_subj + like)
        self.assertEqual(len(m.nodes_db), n_nodes)

    def test_simple_deps(self):
        m = HNodeSimple(self.data, depends_on={'v': 'condition'})
        n_nodes = 2 * (1 + self.n_subj*2) #n_conds * (v_g + n_subj * (v_subj + like))
        self.assertEqual(len(m.nodes_db), n_nodes)

    def test_simplevar_partly_deps(self):
        m = HNodeSimpleVar(self.data, depends_on={'v': 'condition'})
        n_nodes = 1 + 1 + 2 * (1 + self.n_subj*2) #v_std + v_tau + n_conds * (v_g + n_subj * (v_subj + like))
        self.assertEqual(len(m.nodes_db), n_nodes)

    def test_simplevar_deps(self):
        m = HNodeSimpleVar(self.data, depends_on={'v': 'condition', 'v_std':'condition'})
        #n_nodes = n_conds * (v_std + v_tau + v_g + n_subj * (v_subj + like))
        n_nodes = 2 * (1 + 1 + 1 + self.n_subj*2)
        self.assertEqual(len(m.nodes_db), n_nodes)

    def test_simplevar_double_deps_A(self):
        m = HNodeSimpleVar(self.data, depends_on={'v': 'condition', 'v_std':'condition2'})
        #n_nodes = 2*v_tau + 2*v_std + 2*v_g + 4*n_subj*(v_subj + like))
        n_nodes = 2 + 2 + 2 + 4 * self.n_subj * 2
        self.assertEqual(len(m.nodes_db), n_nodes)

    def test_simplevar_double_deps_B(self):
        m = HNodeSimpleVar(self.data, depends_on={'v': ['condition', 'condition2'], 'v_std':'condition2'})
        #n_nodes = 2*v_tau + 2*v_std + 4*v_g + 4*n_subj*(v_subj + like))
        n_nodes = 2 + 2 + 4 + 4 * self.n_subj * 2
        self.assertEqual(len(m.nodes_db), n_nodes)


class TestEstimation(unittest.TestCase):
    """
    simple tests to see if hierarchical merthods do not raise and error
    """

    def test_map_approx(self):
        data, params = kabuki.generate.gen_rand_data(normal_like, {'A':{'loc':0, 'scale':1}, 'B': {'loc':2, 'scale':1}}, samples=100, subjs=100)
        model = HNodeSimple(data, depends_on={'v_g': 'condition'})
        model.approximate_map()

        for condition, subj_params in params.iteritems():
            nodes = model.nodes_db[model.nodes_db['condition'] == condition]
            for idx, params in enumerate(subj_params):
                nodes_subj = nodes[nodes['subj_idx'] == idx]
                for param_name, value in params.iteritems():
                    node = nodes_subj[nodes_subj.name == param_name]
                    assert len(node) == 1, "Only one node should have been left after slicing."
                    np.testing.assert_approx_equal(node.node.value, value, 1)