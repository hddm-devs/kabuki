import kabuki
from kabuki.hierarchical import Knode
import numpy as np

def load_models():
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

def sample_from_models(models, n_iter = 200):
    """sample from all models"""
    for i, model in enumerate(models):
        print "sample model", i
        model.sample(n_iter)
