import kabuki
from kabuki.hierarchical import Knode
import numpy as np
import unittest
import scipy.stats
import kabuki.analyze as ka
from matplotlib.pyplot import close


class TestAnalyze(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        
        #load models
        self.models = self.load_models()
        self.n_models = len(self.models)

        #run models
        print "in init"
        import time
        time.sleep(0.1)

#        kabuki.debug_here()
        self.sample_from_models()
    
    @classmethod
    def load_models(self):
        """
        This function returns a list of models that are going to be tested
        """
        import hddm
        n = 400
        dtype = [('response', np.int), ('rt', np.float), ('subj_idx', np.int32), ('cond1', 'S20'), ('cond2', 'S20')]
        data = np.empty(n, dtype=dtype)
        data['rt'] = np.random.rand(n) + 0.5;
        data['response'] = np.random.randint(2, size=n)
        data['cond1'] = np.array(['A','B'])[np.random.randint(2, size=n)]
        data['cond2'] = np.array(['A','B'])[np.random.randint(2, size=n)]
        data['subj_idx'] = np.zeros(n)

        models = []
        #model 1
        m = hddm.HDDM(data, depends_on = {'v':'cond1'})
        models.append(m)

        #model 2
        m = hddm.HDDM(data, depends_on = {'v':['cond1', 'cond2'], 'a':'cond1'}, include =['z','V'])
        models.append(m)

        data['subj_idx'] = np.random.randint(5, size=n)
        #model 3
        m = hddm.HDDM(data, depends_on = {'v':'cond1'})
        models.append(m)

        #model 4
        v_dict = {'share_var': True}

        #sv has no subj nodes, and it is switched to half-cauchy
        V_g = Knode(kabuki.utils.HalfCauchy, S=10, value=1)
        V = kabuki.Parameter('V', group_knode=V_g,
                             optional=True, default=0)

        m = hddm.HDDM(data, depends_on = {'v':['cond1', 'cond2'], 'a':'cond1'}, include =['V'],
                      update_params = {'v' : v_dict}, replace_params = [V])
        models.append(m)

        return models

    @classmethod
    def sample_from_models(self):
        """sample from all models"""
        n_iters = 200
        for i, model in enumerate(self.models):
            print "sample model", i
            model.sample(n_iters)

    def runTest(self):
        pass

    def test_print_stats(self):
        for model in self.models:
            ka.print_stats(model.stats())
            ka.print_group_stats(model.stats())

    def test_group_plot(self):
        for model in self.models:
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
            ka.group_cond_diff(model,'v', 0, 1)
    
    def test_post_pred_check(self):
        for model in self.models:
            ka.post_pred_check(model, samples=20, bins=100, plot=True, progress_bar=False)
            close('all')
            
    def test_plot_posterior_predictive(self):
        for model in self.models:
            ka.plot_posterior_predictive(model, samples=10)