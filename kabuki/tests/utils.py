import kabuki
from kabuki.hierarchical import Knode
import numpy as np
import pymc as pm
import pandas as pd

class HNodeSimple(kabuki.Hierarchical):
    def create_knodes(self):
        if self.is_group_model:
            mu_g = Knode(pm.Uniform, 'mu_g', lower=-5, upper=5, depends=self.depends['mu'])
            mu_subj = Knode(pm.Normal, 'mu_subj', mu=mu_g, tau=1, depends=('subj_idx',), subj=True)

            like = Knode(pm.Normal, 'like', mu=mu_subj, tau=1, col_name='data', observed=True)

            return [mu_g, mu_subj, like]

        else:
            mu_subj = Knode(pm.Uniform, 'mu_subj', lower=-5, upper=5, depends=self.depends['mu'])

            like = Knode(pm.Normal, 'like', mu=mu_subj, tau=1, col_name='data', observed=True)

            return [mu_subj, like]

class HNodeSimpleVar(kabuki.Hierarchical):
    def create_knodes(self):
        if self.is_group_model:
            mu_g = Knode(pm.Uniform, 'mu_g', lower=-5, upper=5, depends=self.depends['mu'])
            mu_std = Knode(pm.Uniform, 'mu_std', lower=1e-8, upper=100, depends=self.depends['mu_std'])
            mu_tau = Knode(pm.Deterministic, 'mu_tau', doc='mu_tau', eval=lambda x: x**-2, x=mu_std, plot=False, trace=False)
            #v_tau = Knode(pm.Lambda, 'v_tau', lam_fun=lambda x=v_std: x**-2, plot=False, trace=False)
            mu_subj = Knode(pm.Normal, 'mu_subj', mu=mu_g, tau=mu_tau, subj=True)

            like = Knode(pm.Normal, 'like', mu=mu_subj, tau=1, col_name='data', observed=True)

            return [mu_g, mu_std, mu_tau, mu_subj, like]

        else:
            mu_subj = Knode(pm.Uniform, 'mu_subj', lower=-5, upper=5, depends=self.depends['mu'])

            like = Knode(pm.Normal, 'like', mu=mu_subj, tau=1, col_name='data', observed=True)

            return [mu_subj, like]

class HNodeTransform(kabuki.Hierarchical):
    def create_knodes(self):
        if self.is_group_model:
            mu_g = Knode(pm.Uniform, 'mu_g', lower=-5, upper=5, depends=self.depends['mu'], hidden=True)
            mu_std = Knode(pm.Uniform, 'mu_std', lower=1e-8, upper=100, depends=self.depends['mu_std'])
            mu_tau = Knode(pm.Deterministic, 'mu_tau', doc='mu_tau', eval=lambda x: x**-2, x=mu_std, plot=False, trace=False)
            mu_subj = Knode(pm.Normal, 'mu_subj', mu=mu_g, tau=mu_tau, subj=True, plot=False, hidden=True)
            mu_subj_trans = Knode(pm.Deterministic, 'mu_subj_trans', eval=lambda x: x, x=mu_subj, plot=True, trace=True)
            like = Knode(pm.Normal, 'like', mu=mu_subj_trans, tau=1, col_name='data', observed=True)

            return [mu_g, mu_std, mu_tau, mu_subj, mu_subj_trans, like]

        else:
            mu_subj = Knode(pm.Uniform, 'mu_subj', lower=-5, upper=5, depends=self.depends['mu'])
            mu_subj_trans = Knode(pm.Deterministic, 'mu_subj_trans', eval=lambda x: x, x=mu_subj, plot=True, trace=True)

            like = Knode(pm.Normal, 'like', mu=mu_subj_trans, tau=1, col_name='data', observed=True)

            return [mu_subj, mu_subj_trans, like]

def gen_func_df(size=100, loc=0, scale=1):
    data = np.random.normal(loc=loc, scale=scale, size=size)
    return pd.DataFrame(data, columns=['data'])


def create_test_models():
    n_subj = 5
    data, params = kabuki.generate.gen_rand_data(gen_func_df, {'A':{'loc':0, 'scale':1}, 'B': {'loc':0, 'scale':1}}, subjs=n_subj, size=20)
    data = pd.DataFrame(data)
    data['condition2'] = np.random.randint(2, size=len(data))
    models = []

    models.append(HNodeSimple(data))
    models.append(HNodeSimple(data, depends_on={'mu': 'condition'}))
    models.append(HNodeSimpleVar(data, depends_on={'mu': 'condition'}))
    models.append(HNodeSimpleVar(data, depends_on={'mu': 'condition'}, is_group_model=False))
    models.append(HNodeSimpleVar(data, depends_on={'mu': 'condition', 'mu_std':'condition'}))
    models.append(HNodeSimpleVar(data, depends_on={'mu': 'condition', 'mu_std':'condition2'}))
    models.append(HNodeSimpleVar(data, depends_on={'mu': ['condition', 'condition2'], 'mu_std':'condition2'}))
    models.append(HNodeTransform(data, depends_on={'mu': 'condition', 'mu_std':'condition2'}))

    return models, params

def sample_from_models(models, n_iter = 200):
    """sample from all models"""
    for i, model in enumerate(models):
        print("sample model", i)
        model.sample(n_iter)
