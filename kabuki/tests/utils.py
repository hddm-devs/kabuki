import kabuki
from kabuki.hierarchical import Knode
import numpy as np
import pymc as pm
import scipy
import pandas as pd

normal_like = kabuki.distributions.scipy_stochastic(scipy.stats.distributions.norm_gen, name='normal', longname='normal')

class HNodeSimple(kabuki.Hierarchical):
    def create_knodes(self):
        v_g = Knode(pm.Uniform, 'v_g', lower=-5, upper=5, depends=self.depends['v'])
        v_subj = Knode(pm.Normal, 'v_subj', mu=v_g, tau=1, depends=('subj_idx',), subj=True)

        like = Knode(pm.Normal, 'like', mu=v_subj, tau=1, col_name='data', observed=True)

        return [v_g, v_subj, like]

    def create_knodes_single_subj(self):
        v_subj = Knode(pm.Uniform, 'v_subj', lower=-5, upper=5, depends=self.depends['v'])

        like = Knode(pm.Normal, 'like', mu=v_subj, tau=1, col_name='data', observed=True)

        return [v_subj, like]

class HNodeSimpleVar(kabuki.Hierarchical):
    def create_knodes(self):
        v_g = Knode(pm.Uniform, 'v_g', lower=-5, upper=5, depends=self.depends['v'])
        v_std = Knode(pm.Uniform, 'v_std', lower=1e-8, upper=100, depends=self.depends['v_std'])
        v_tau = Knode(pm.Deterministic, 'v_tau', doc='v_tau', eval=lambda x: x**-2, x=v_std, plot=False, trace=False)
        #v_tau = Knode(pm.Lambda, 'v_tau', lam_fun=lambda x=v_std: x**-2, plot=False, trace=False)
        v_subj = Knode(pm.Normal, 'v_subj', mu=v_g, tau=v_tau, subj=True)

        like = Knode(pm.Normal, 'like', mu=v_subj, tau=1, col_name='data', observed=True)

        return [v_g, v_std, v_tau, v_subj, like]

    def create_knodes_single_subj(self):
        v_subj = Knode(pm.Uniform, 'v_subj', lower=-5, upper=5, depends=self.depends['v'])

        like = Knode(pm.Normal, 'like', mu=v_subj, tau=1, col_name='data', observed=True)

        return [v_subj, like]

def create_test_models():
    n_subj = 3
    data = kabuki.generate.gen_rand_data(normal_like, {'A':{'loc':0, 'scale':1}, 'B': {'loc':0, 'scale':1}}, subjs=n_subj)[0]
    data = pd.DataFrame(data)
    data['condition2'] = np.random.randint(2, size=len(data))
    models = []

    models.append(HNodeSimple(data['data']))
    models.append(HNodeSimple(data))
    models.append(HNodeSimple(data, depends_on={'v': 'condition'}))
    models.append(HNodeSimpleVar(data, depends_on={'v': 'condition'}))
    models.append(HNodeSimpleVar(data, depends_on={'v': 'condition', 'v_std':'condition'}))
    models.append(HNodeSimpleVar(data, depends_on={'v': 'condition', 'v_std':'condition2'}))
    models.append(HNodeSimpleVar(data, depends_on={'v': ['condition', 'condition2'], 'v_std':'condition2'}))

    return models

def sample_from_models(models, n_iter = 200):
    """sample from all models"""
    for i, model in enumerate(models):
        print "sample model", i
        model.sample(n_iter)
