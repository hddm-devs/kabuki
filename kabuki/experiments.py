import numpy as np
import kabuki
import pymc as pm
import os.path
from copy import deepcopy
import matplotlib.pyplot as plt

def sort_dict(d):
    from operator import itemgetter
    return sorted(d.iteritems(), key=itemgetter(1))


def _model_type_to_class(model_type):
    if isinstance(model_type, kabuki.Hierarchical):
        model_class = model_type
    elif isinstance(model_type, str):
        model_class = kabuki.utils.find_object(model_type)
    else:
        raise TypeError('Model type %s not supported.' % type(model_type))

    return model_class


def _parse_experiment(experiment):
    data = experiment['data']
    model_type = experiment['model_type']

    if 'kwargs' in experiment:
        kwargs = deepcopy(experiment['kwargs'])
    else:
        kwargs = {}


    if 'name' in experiment:
        name = experiment['name']
    else:
        name = model_type + str(sort_dict(kwargs))

    model_class = _model_type_to_class(model_type)

    return data, model_class, kwargs, name


def run_experiment(experiment, db='sqlite', samples=10000, burn=5000, thin=3, subj_map_init=True):
    import os

    data, model_class, kwargs, name = _parse_experiment(experiment)

    if not os.path.exists(name):
        os.mkdir(name)

    m = model_class(data, **kwargs)

    if subj_map_init:
        m.subj_by_subj_map_init()

    m.mcmc(db=db, dbname=os.path.join(name, 'traces.db'))
    m.sample(samples, burn=burn, thin=thin)

    stats = kabuki.analyze.gen_stats(m.mc.stats())

    with open('%s/stats.txt'%name, 'w') as f:
        f.write("%f\n" % m.mc.dic)
        f.write(stats)

    return stats


def run_experiments(experiments, mpi=False, **kwargs):
    if mpi:
        import mpi4py_map
        results = mpi4py_map.map(run_experiment, experiments, **kwargs)
    else:
        results = [run_experiment(experiment, **kwargs) for experiment in experiments]

    return results


def load_experiment(experiment, db_loader=pm.database.sqlite.load, db_fname='traces.db'):
    experiment['model'] = load_model(experiment, db_loader=db_loader, db_fname=db_fname)
    return experiment

def load_model(experiment, db_loader=pm.database.sqlite.load, db_fname='traces.db'):
    data, model_class, kwargs, name = _parse_experiment(experiment)
    m = model_class(data, **kwargs)

    m.load_db(os.path.join(name, db_fname), db_loader=db_loader)

    return m

def load_ppc(experiment):
    import pandas as pd
    data, model_class, kwargs, name = _parse_experiment(experiment)
    post_pred = pd.read_csv(os.path.join(name, 'post_pred.csv'))

    return post_pred

def load_ppcs(experiments):
    import pandas as pd
    post_preds = []
    for experiment in experiments:
        try:
            ppc = load_ppc(experiment)
        except:
            pass
        post_preds.append(ppc)

    model_names = [_parse_experiment(experiment)[-1] for experiment in experiments]
    print model_names
    return (model_names, post_preds)
    #return pd.concat(post_preds, keys=model_names, names=['model'])

def analyze_experiment(experiment, plot_groups=True, plot_traces=True, plot_post_pred=True, ppd=True, stats=None):
    data, model_class, kwargs, name = _parse_experiment(experiment)

    if 'model' in experiment:
        # Model already loaded, use it
        model = experiment['model']
    else: # Load it
        model = load_model(experiment)

    print "Analyzing model: %s" % name
    if plot_groups:
        kabuki.analyze.group_plot(model, save_to=name)

    if plot_traces:
        model.plot_posteriors(path=name)

    if plot_post_pred:
        print "Plotting posterior predictive"
        kabuki.analyze.plot_posterior_predictive(model, np.linspace(-1.2, 1.2, 80), savefig=True, prefix=name, columns=7, figsize=(18,18))

    if ppd:
        ppd = kabuki.analyze.post_pred_check(model, stats=stats)
        print ppd
        ppd.to_csv(os.path.join(name, 'post_pred.csv'))


def analyze_experiments(experiments, mpi=False, plot_dic=True, **kwargs):
    if mpi:
        import mpi4py_map
        results = mpi4py_map.map(analyze_experiment, experiments, **kwargs)
    else:
        results = [analyze_experiment(experiment, **kwargs) for experiment in experiments]

    if plot_dic:
        dics = [experiment['model'].mc.dic for experiment in experiments]
        names = [_parse_experiment(experiment)[-1] for experiment in experiments]

        fig = plt.figure()
        x = range(len(names))
        ax = plt.bar(x, dics, align='center')
        plt.xticks(x, names)
        plt.ylabel('DIC')
        fig.autofmt_xdate()
        fig.savefig('dic.png')
        fig.savefig('dic.pdf')

    return results
