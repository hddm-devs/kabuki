import kabuki
import numpy as np
import unittest
import pymc as pm
from utils import HNodeSimple, HNodeSimpleVar, normal_like, sample_from_models, create_test_models

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
                try:
                    model.map()
                    raise ValueError("NotimplementedError should have been raised")
                except NotImplementedError:
                    pass
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


