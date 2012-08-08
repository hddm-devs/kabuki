import numpy as np
import unittest
import kabuki.analyze as ka
from matplotlib.pyplot import close
import test_utils

class TestAnalyzeBreakdown(unittest.TestCase):
    """
    test unit for analyze.py
    the unit only tests to see if the functions donot raise an error.
    it does not check the validity of the results.
    """

    @classmethod
    def setUpClass(self):

        #load models
        self.models = test_utils.load_models()

        #run models
        test_utils.sample_from_models(self.models, n_iter=200)

    def runTest(self):
        pass

    def test_group_plot(self):
        for model in self.models:
            if model.is_group_model:
                ka.group_plot(model)
                close('all')

    def test_plot_posteriors_nodes(self):
        for model in self.models:
            ka.plot_posterior_nodes(model.mc.stochastics, bins=50)
            close('all')

    def test_compare_all_pairwise(self):
        for model in self.models:
            ka.compare_all_pairwise(model)

    def plot_all_pairwise(self):
        for model in self.models:
            ka.plot_all_pairwise(model)

    @unittest.skip("Not implemented")
    def test_savage_dickey(self):
        raise NotImplementedError

    @unittest.skip("Not implemented")
    def test_gelman_rubin(self):
        raise NotImplementedError

    @unittest.skip("Not implemented")
    def test_check_geweke(self):
        raise NotImplementedError

    def test_group_cond_diff(self):
        for model in self.models:
            if model.is_group_model:
                if model.depends:
                    (name, cond) = model.depends.items()[0]
                    tags = model.params_dict[name].group_nodes.keys()[:2]
                ka.group_cond_diff(model,name, *tags)

    def test_post_pred_check(self):
        for model in self.models:
            ka.post_pred_check(model, samples=20, bins=100, plot=True, progress_bar=False)
            close('all')

    def test_plot_posterior_predictive(self):
        for model in self.models:
            ka.plot_posterior_predictive(model, value_range = np.arange(-2,2,10), samples=10)