
import numpy as np
import unittest
import kabuki.analyze as ka
from matplotlib.pyplot import close
import utils

class TestAnalyzeBreakdown(unittest.TestCase):
    """
    test unit for analyze.py
    the unit only tests to see if the functions donot raise an error.
    it does not check the validity of the results.
    """

    @classmethod
    def setUpClass(self):

        #load models
        self.models, _ = utils.create_test_models()

        #run models
        utils.sample_from_models(self.models, n_iter=200)

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

    @unittest.skip("Not implemented")
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

    @unittest.skip("Not implemented")
    def test_group_cond_diff(self):
        for model in self.models:
            if model.is_group_model:
                if model.depends:
                    (name, cond) = model.depends.items()[0]
                    tags = model.nodes_db[name].group_nodes.keys()[:2]
                ka.group_cond_diff(model,name, *tags)

    @unittest.skip("Fails because of pymc likelihoods converting DataFrames to numpy arrays.")
    def test_post_pred_check(self):
        for model in self.models:
            ka.post_pred_gen(model, samples=20, progress_bar=False)

    def test_plot_posterior_predictive(self):
        for model in self.models:
            ka.plot_posterior_predictive(model, value_range=np.arange(-2,2,10), samples=10)
