import kabuki
from kabuki.hierarchical import Knode
import numpy as np
import unittest
import pymc as pm
import test_utils
import scipy
import pandas as pd

normal_like = kabuki.distributions.scipy_stochastic(scipy.stats.distributions.norm_gen, name='normal', longname='normal')

class HNodeSimple(kabuki.Hierarchical):
    def create_knodes(self):
        v_g = Knode(pm.Uniform, 'v_g', lower=-5, upper=5, depends=self.depends['v'])
        v_subj = Knode(pm.Normal, 'v_subj', mu=v_g, tau=1, depends=('subj_idx',), subj=True)

        like = Knode(pm.Normal, 'like', mu=v_subj, tau=1, col_name='data', observed=True)

        return [v_g, v_subj, like]

class HNodeSimpleVar(kabuki.Hierarchical):
    def create_knodes(self):
        v_g = Knode(pm.Uniform, 'v_g', lower=-5, upper=5, depends=self.depends['v'])
        v_std = Knode(pm.Uniform, 'v_std', lower=1e-8, upper=100, depends=self.depends['v_std'])
        v_tau = Knode(pm.Lambda, 'v_tau', lam_fun=lambda x=v_std: x**-2, plot=False, trace=False)
        v_subj = Knode(pm.Normal, 'v_subj', mu=v_g, tau=v_tau, subj=True)

        like = Knode(pm.Normal, 'like', mu=v_subj, tau=1, col_name='data', observed=True)

        return [v_g, v_tau, v_subj, like]

class TestModels(unittest.TestCase):

    #@classmethod
    def setUp(self):
        data = kabuki.generate.gen_rand_data(normal_like, {'A':{'loc':0, 'scale':1}, 'B': {'loc':0, 'scale':1}}, subjs=3)[0]
        data = pd.DataFrame(data)
        data['condition2'] = np.random.randint(2, size=len(data))
        self.data = data

    def runTest(self):
        pass

    def test_simple_no_deps(self):
        m = HNodeSimple(self.data)

    def test_simple_deps(self):
        m = HNodeSimple(self.data, depends_on={'v': 'condition'})

    def test_simplevar_partly_deps(self):
        m = HNodeSimpleVar(self.data, depends_on={'v': 'condition'})

    def test_simplevar_deps(self):
        m = HNodeSimpleVar(self.data, depends_on={'v': 'condition', 'v_std':'condition'})

    def test_simplevar_double_deps(self):
        m = HNodeSimpleVar(self.data, depends_on={'v': 'condition', 'v_std':'condition2'})

    def test_simplevar_double_deps(self):
        m = HNodeSimpleVar(self.data, depends_on={'v': ['condition', 'condition2'], 'v_std':'condition2'})

