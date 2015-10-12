import numpy as np
import kabuki
import pymc as pm
import os.path
from copy import deepcopy
import matplotlib.pyplot as plt

def sort_dict(d):
    from operator import itemgetter
    return sorted(iter(d.items()), key=itemgetter(1))


def _model_type_to_class(model_type):
    """Takes either a (inherited) class of Hierarchical or a str with
    the name of the class and returns the class.

    """
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
    """Run a single experiment: Builds the model, initializes,
    samples. See analyze_experiment() for how to generate output
    statistics of your finished model run.

    :Arguments:
        experiment : dict
            dict containing the following mappings:
            * 'data' -> np.ndarray
            * 'model_type' -> Hierarchical class or str name of class
            * 'name' -> str of name of experiment (optional)
            * 'kwargs' -> Keyword arguments to be supplied at model creation
        db : str (default='sqlite')
            Which database backend to use
        samples : int (default=10000)
            How many posterior samples to draw
        burn : int (default=5000)
            How many posterior samples to throw away as burn-in
        thin : int (default=3)
            How much thinning to apply
        subj_map_init : bool (default=True)
            Whether to initialize the model using subj_by_subj_map_init().

    :Returns:
        Str of summary statistics.

    """
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
    """Run a list of experiments. Optionally using MPI.

    :Arguments:
        experiments : list of dicts
            One entry for each experiment to run. Each experiment should have a dict like this:
            * 'data' -> np.ndarray
            * 'model_type' -> Hierarchical class or str name of class
            * 'name' -> str of name of experiment (optional)
            * 'kwargs' -> Keyword arguments to be supplied at model creation
        mpi : bool (default=False)
            Whether to run experiments in parallel using MPI. If set to True you must call
            your script (which calls this function) using mpirun.
            This requires mpi4py_map and mpi4py to be installed.
            E.g. pip install mpi4py mpi4py_map
        kwargs : dict
            Additional keyword arguments to be passed to run_experiment (see run_experiment),
            such as samples, burn etc.
    """
    if mpi:
        import mpi4py_map
        results = mpi4py_map.map(run_experiment, experiments, **kwargs)
    else:
        results = [run_experiment(experiment, **kwargs) for experiment in experiments]

    return results


def load_experiment(experiment, dbname='traces.db', db='sqlite'):
    """Load specific experiment from a database file.

    :Arguments:
        experiment : dict (see run_experiment for the contents)
        dbname : str (default='traces.db')
            Name of the database file.
        db : str (default='sqlite')
            Which database backend to use, can be
            sqlite, pickle, hdf5, txt.

    :Return:
        Same experiment with a new 'model' key linking to the loaded model.
    """

    experiment['model'] = load_model(experiment, db=db, dbname=dbname)
    return experiment

def load_model(experiment, db='sqlite', dbname='traces.db'):
    data, model_class, kwargs, name = _parse_experiment(experiment)
    m = model_class(data, **kwargs)

    m.load_db(os.path.join(name, dbname), db='sqlite')

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
    print(model_names)
    return (model_names, post_preds)
    #return pd.concat(post_preds, keys=model_names, names=['model'])

def analyze_experiment(experiment, plot_groups=True, plot_traces=True, plot_post_pred=True, ppc=True, stats=None):
    """Analyze a single experiment. Writes output statitics and various plots into a subdirectory.

    :Arguments:
        experiment : dict
            Should be a dict like this:
            * 'data' -> np.ndarray
            * 'model_type' -> Hierarchical class or str name of class
            * 'name' -> str of name of experiment (optional)
            * 'kwargs' -> Keyword arguments to be supplied at model creation
        plot_groups : bool (default=True)
            Whether to plot all traces in a group plot.
        plot_traces : bool (default=True)
            Whether to plot all traces in a trace plot.
        plot_post_pred : bool (default=True)
            Whether to plot the posterior predictive.
        ppc : bool (default=True)
            Whether to run a posterior predictive check.
        stats : dict (default=None)
            The stats to be used for the posterior predictive check (see kabuki.analyze.post_pred_check)
    """
    data, model_class, kwargs, name = _parse_experiment(experiment)

    if 'model' in experiment:
        # Model already loaded, use it
        model = experiment['model']
    else: # Load it
        model = load_model(experiment)

    print("Analyzing model: %s" % name)
    if plot_groups:
        kabuki.analyze.group_plot(model, save_to=name)

    if plot_traces:
        model.plot_posteriors(path=name)

    if plot_post_pred:
        print("Plotting posterior predictive")
        kabuki.analyze.plot_posterior_predictive(model, np.linspace(-1.2, 1.2, 80), savefig=True, path=name, columns=7, figsize=(18,18))

    if ppc:
        ppc = kabuki.analyze.post_pred_check(model, stats=stats)
        print(ppc)
        ppc.to_csv(os.path.join(name, 'post_pred.csv'))


def analyze_experiments(experiments, mpi=False, plot_dic=True, **kwargs):
    """Analyze multiple experiments. Outputs will be saved to
    subdirectories. Can optionally analyze in parallel.

    :Arguments:
        experiments : list of dicts
            One entry for each experiment to analyze. Each experiment should have a dict like this:
            * 'data' -> np.ndarray
            * 'model_type' -> Hierarchical class or str name of class
            * 'name' -> str of name of experiment (optional)
            * 'kwargs' -> Keyword arguments to be supplied at model creation
        mpi : bool (default=False)
            Whether to run experiments in parallel using MPI. If set to True you must call
            your script (which calls this function) using mpirun.
            This requires mpi4py_map and mpi4py to be installed.
            E.g. pip install mpi4py mpi4py_map
        plot_dic : bool (default=True)
            Whether to create a bar plot of DIC values for each model.
        kwargs : dict
            Keyword arguments to be passed to analyze_experiment. See analyze_experiment.

    """

    # Load models if necessary
    for experiment in experiments:
        if 'model' not in experiment:
            experiment['model'] = load_model(experiment)

    if mpi:
        import mpi4py_map
        results = mpi4py_map.map(analyze_experiment, experiments, **kwargs)
    else:
        results = [analyze_experiment(experiment, **kwargs) for experiment in experiments]

    if plot_dic:
        dics = [experiment['model'].mc.dic for experiment in experiments]
        names = [_parse_experiment(experiment)[-1] for experiment in experiments]

        fig = plt.figure()
        x = list(range(len(names)))
        ax = plt.bar(x, dics, align='center')
        plt.xticks(x, names)
        plt.ylabel('DIC')
        fig.autofmt_xdate()
        fig.savefig('dic.png')
        fig.savefig('dic.pdf')

    return results

if __name__=='__main__':
    from copy import copy

    # Example, requires HDDM
    import hddm

    # Generate data
    params_single = hddm.generate.gen_rand_params()
    params = {'cond1': copy(params_single), 'cond2': copy(params_single)}
    params['cond2']['v'] = 0
    data, subj_params = kabuki.generate.gen_rand_data(hddm.likelihoods.Wfpt,
                                         params=params, samples=100, subjs=10, column_name='rt')[0]

    # Create different models to test our various hypotheses.
    experiments = [
                   {'name': 'baseline', 'data': data, 'model_type': 'hddm.HDDM', 'kwargs': {'depends_on': {}}},
                   {'name': 'condition_influences_drift', 'data': data, 'model_type': 'hddm.HDDM', 'kwargs': {'depends_on': {'v': 'condition'}}},
                   {'name': 'condition_influences_threshold', 'data': data, 'model_type': 'hddm.HDDM', 'kwargs': {'depends_on': {'a': 'condition'}}}]

    print("Running experiments...")
    run_experiments(experiments)

    print("Analyzing experiments...")
    analyze_experiments(experiments, ppc=False)

    print("Done! Check the newly created subdirectories.")


