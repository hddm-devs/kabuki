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
        assert(len(m.nodes) == n_nodes)

    def test_simple_deps(self):
        m = HNodeSimple(self.data, depends_on={'v': 'condition'})
        n_nodes = 2 * (1 + self.n_subj*2) #n_conds * (v_g + n_subj * (v_subj + like))
        assert(len(m.nodes) == n_nodes)

    def test_simplevar_partly_deps(self):
        m = HNodeSimpleVar(self.data, depends_on={'v': 'condition'})
        n_nodes = 1 + 2 * (1 + self.n_subj*2) #v_std + n_conds * (v_g + n_subj * (v_subj + like))
        assert(len(m.nodes) == n_nodes)

    def test_simplevar_deps(self):
        m = HNodeSimpleVar(self.data, depends_on={'v': 'condition', 'v_std':'condition'})
        #n_nodes = n_conds * (v_std + v_g + n_subj * (v_subj + like))
        n_nodes = 2 * (2 + self.n_subj*2)
        assert(len(m.nodes) == n_nodes)

    def test_simplevar_double_deps_A(self):
        m = HNodeSimpleVar(self.data, depends_on={'v': 'condition', 'v_std':'condition2'})
        #n_nodes = 2*v_std + 2*v_g + 4*n_subj*(v_subj + like))
        n_nodes = 2 + 2 + 4 * self.n_subj * 2
        assert(len(m.nodes) == n_nodes)

    def test_simplevar_double_deps_B(self):
        m = HNodeSimpleVar(self.data, depends_on={'v': ['condition', 'condition2'], 'v_std':'condition2'})
        #n_nodes = 2*v_std + 4*v_g + 4*n_subj*(v_subj + like))
        n_nodes = 2 + 4 + 4 * self.n_subj * 2
        assert(len(m.nodes) == n_nodes)

