import kabuki
from kabuki.hierarchical import Knode
import numpy as np
import pymc as pm
import scipy
import pandas as pd

normal_like = kabuki.distributions.scipy_stochastic(scipy.stats.distributions.norm_gen, name='normal', longname='normal')

class HNodeSimple(kabuki.Hierarchical):
    def create_knodes(self):
        loc_g = Knode(pm.Uniform, 'loc_g', lower=-5, upper=5, depends=self.depends['loc'])
        loc_subj = Knode(pm.Normal, 'loc_subj', mu=loc_g, tau=1, depends=('subj_idx',), subj=True)

        like = Knode(pm.Normal, 'like', mu=loc_subj, tau=1, col_name='data', observed=True)

        return [loc_g, loc_subj, like]

    def create_knodes_single_subj(self):
        loc_subj = Knode(pm.Uniform, 'loc_subj', lower=-5, upper=5, depends=self.depends['loc'])

        like = Knode(pm.Normal, 'like', mu=loc_subj, tau=1, col_name='data', observed=True)

        return [loc_subj, like]

class HNodeSimpleVar(kabuki.Hierarchical):
    def create_knodes(self):
        loc_g = Knode(pm.Uniform, 'loc_g', lower=-5, upper=5, depends=self.depends['loc'])
        loc_std = Knode(pm.Uniform, 'loc_std', lower=1e-8, upper=100, depends=self.depends['loc_std'])
        loc_tau = Knode(pm.Deterministic, 'loc_tau', doc='loc_tau', eval=lambda x: x**-2, x=loc_std, plot=False, trace=False)
        #v_tau = Knode(pm.Lambda, 'v_tau', lam_fun=lambda x=v_std: x**-2, plot=False, trace=False)
        loc_subj = Knode(pm.Normal, 'loc_subj', mu=loc_g, tau=loc_tau, subj=True)

        like = Knode(pm.Normal, 'like', mu=loc_subj, tau=1, col_name='data', observed=True)

        return [loc_g, loc_std, loc_tau, loc_subj, like]

    def create_knodes_single_subj(self):
        loc_subj = Knode(pm.Uniform, 'loc_subj', lower=-5, upper=5, depends=self.depends['loc'])

        like = Knode(pm.Normal, 'like', mu=loc_subj, tau=1, col_name='data', observed=True)

        return [loc_subj, like]

def create_test_models():
    n_subj = 5
    data, params = kabuki.generate.gen_rand_data(normal_like, {'A':{'loc':0, 'scale':1}, 'B': {'loc':0, 'scale':1}}, subjs=n_subj)
    data = pd.DataFrame(data)
    data['condition2'] = np.random.randint(2, size=len(data))
    models = []

    models.append(HNodeSimple(data['data']))
    models.append(HNodeSimple(data))
    models.append(HNodeSimple(data, depends_on={'loc': 'condition'}))
    models.append(HNodeSimpleVar(data, depends_on={'loc': 'condition'}))
    models.append(HNodeSimpleVar(data, depends_on={'loc': 'condition', 'loc_std':'condition'}))
    models.append(HNodeSimpleVar(data, depends_on={'loc': 'condition', 'loc_std':'condition2'}))
    models.append(HNodeSimpleVar(data, depends_on={'loc': ['condition', 'condition2'], 'loc_std':'condition2'}))

    return models, params

def sample_from_models(models, n_iter = 200):
    """sample from all models"""
    for i, model in enumerate(models):
        print "sample model", i
        model.sample(n_iter)
