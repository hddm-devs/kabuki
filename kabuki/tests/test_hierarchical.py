import os
import kabuki
import numpy as np
import unittest
from nose.tools import raises
import pymc as pm
from utils import HNodeSimple, HNodeSimpleVar, sample_from_models, create_test_models
import pandas as pd

from utils import gen_func_df

class TestHierarchicalBreakDown(unittest.TestCase):
    """
    simple tests to see if hierarchical merthods do not raise and error
    """

    @classmethod
    def setUpClass(self):
    #def setUp(self):

        #load models
        self.models, self.params = create_test_models()

        #run models
        sample_from_models(self.models, n_iter=500)

    def runTest(self):
        pass

    def test_map(self):
        np.random.seed(123)
        for model in self.models:
            model.find_starting_values()

    def test_emcee(self):
        self.models, self.params = create_test_models()
        np.random.seed(123)
        for model in self.models:
            model.find_starting_values()
            model.sample_emcee()


    def test_dic_info(self):
        for model in self.models:
            model.dic_info

    def test_print_stats(self):
        for model in self.models:
            model.print_stats()
            model.print_stats(subj=False)
            if model.is_group_model:
                model.print_stats(subj_idx=1)

    def test_load_db(self):
        new_models, params = create_test_models()
        for i, model in enumerate(new_models):
            print "sample model", i
            model.sample(100, dbname='unittest.db', db='pickle')
            model.load_db(dbname='unittest.db', db='pickle')
            model.gen_stats()
            os.remove('unittest.db')

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

    @classmethod
    def setUpClass(self):
#    def setUp(self):
        self.n_subj = 3
        data, _ = kabuki.generate.gen_rand_data(gen_func_df, {'A':{'loc':0, 'scale':1}, 'B': {'loc':0, 'scale':1}},
                                                subjs=self.n_subj)
        data = pd.DataFrame(data)
        data['condition2'] = np.random.randint(2, size=len(data))
        self.data = data

    def runTest(self):
        pass

    def test_save_and_load_breakdown(self):
        m = HNodeSimple(self.data)
        m.sample(500, dbname='test.db', db='pickle')
        m.save('test.model')
        m_load = kabuki.utils.load('test.model')
        os.remove('test.db')
        os.remove('test.model')

    def test_simple_no_deps(self):
        m = HNodeSimple(self.data)
        n_nodes = 1 + self.n_subj*2 #mu_g + n_subj * (mu_subj + like)
        self.assertEqual(len(m.nodes_db), n_nodes)

    def test_simple_deps(self):
        m = HNodeSimple(self.data, depends_on={'mu': 'condition'})
        n_nodes = 2 * (1 + self.n_subj*2) #n_conds * (mu_g + n_subj * (mu_subj + like))
        self.assertEqual(len(m.nodes_db), n_nodes)

    @raises(AssertionError)
    def test_assertion_on_wrong_param_name(self):
        HNodeSimple(self.data, depends_on={'non_existant': 'condition'})

    def test_assertion_on_wrong_param_name_tofix(self):
        # does not catch if correct argument
        HNodeSimple(self.data, depends_on={'non_existant': 'condition', 'mu': 'condition'})

    def test_simplevar_partly_deps(self):
        m = HNodeSimpleVar(self.data, depends_on={'mu': 'condition'})
        n_nodes = 1 + 1 + 2 * (1 + self.n_subj*2) #mu_std + mu_tau + n_conds * (mu_g + n_subj * (mu_subj + like))
        self.assertEqual(len(m.nodes_db), n_nodes)

    def test_simplevar_deps(self):
        m = HNodeSimpleVar(self.data, depends_on={'mu': 'condition', 'mu_std':'condition'})
        #n_nodes = n_conds * (mu_std + mu_tau + mu_g + n_subj * (mu_subj + like))
        n_nodes = 2 * (1 + 1 + 1 + self.n_subj*2)
        self.assertEqual(len(m.nodes_db), n_nodes)

    def test_simplevar_double_deps_A(self):
        m = HNodeSimpleVar(self.data, depends_on={'mu': 'condition', 'mu_std':'condition2'})
        #n_nodes = 2*v_tau + 2*v_std + 2*v_g + 4*n_subj*(v_subj + like))
        n_nodes = 2 + 2 + 2 + 4 * self.n_subj * 2
        self.assertEqual(len(m.nodes_db), n_nodes)

    def test_simplevar_double_deps_B(self):
        m = HNodeSimpleVar(self.data, depends_on={'mu': ['condition', 'condition2'], 'mu_std':'condition2'})
        #n_nodes = 2*mu_tau + 2*mu_std + 4*mu_g + 4*n_subj*(mu_subj + like))
        n_nodes = 2 + 2 + 4 + 4 * self.n_subj * 2
        self.assertEqual(len(m.nodes_db), n_nodes)


class TestEstimation(unittest.TestCase):
    """
    simple tests to see if hierarchical methods do not raise and error
    """

    def test_map_approx(self):
        subjs = 40
        data, params_true = kabuki.generate.gen_rand_data(gen_func_df,
                                                          {'A':{'loc':0, 'scale':1}, 'B': {'loc':2, 'scale':1}},
                                                          subj_noise={'loc':.1}, size=1000, subjs=subjs)

        model = HNodeSimple(data, depends_on={'mu': 'condition'})

        model.approximate_map()

        counter = 0
        for condition, subj_params in params_true.iteritems():
            nodes = model.nodes_db[model.nodes_db['condition'] == condition]
            for idx, params in enumerate(subj_params):
                nodes_subj = nodes[nodes['subj_idx'] == idx]
                for param_name, value in params.iteritems():
                    if param_name != 'loc':
                        continue
                    node = nodes_subj[nodes_subj.knode_name == 'mu_subj']
                    assert len(node) == 1, "Only one node should have been left after slicing."
                    abs_error = np.abs(node.map[0] - value)
                    self.assertTrue(abs_error < .2)
                    counter += 1

        # looping for each condition (i.e. twice)
        self.assertEqual(counter, subjs*2)


class TestConcatenate(unittest.TestCase):
    def test_concat(self):
        n_subj = 5
        data, params = kabuki.generate.gen_rand_data(gen_func_df, {'A':{'loc':0, 'scale':1}, 'B': {'loc':0, 'scale':1}}, subjs=n_subj)
        data = pd.DataFrame(data)

        models = []
        for i in range(4):
            m = HNodeSimple(data)
            m.sample(100, burn=0, db='pickle', dbname='test_%d'%i)
            models.append(m)

        super_model = kabuki.utils.concat_models(models)
        stochs = super_model.get_stochastics()
        for stoch in stochs.node:
            self.assertEqual(len(stoch.trace[:]), 100*4)

        for i in range(4):
            os.remove('test_%d'%i)
