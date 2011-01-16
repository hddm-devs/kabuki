#!/usr/bin/python
from __future__ import division
import numpy as np
import numpy.lib.recfunctions as rec
import pymc as pm
from copy import copy

import kabuki

def scale(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

# Model classes
class Base(object):
    """Base class for MCMC models."""
    def __init__(self):
        self.model = None
        self.mcmc_model = None
        self.map_model = None
        self.params_est = {}
        self.params_est_std = {}
        self.params_est_perc = {}
        self.stats = {}
	self.colors = ('r','b','g','y','c')

    def _set_params(self):
        raise NotImplementedError("This method has to be overloaded")
    
    def _set_model(self):
        raise NotImplementedError("This method has to be overloaded")

    def _set_all_params(self):
        self._set_params()


    def map(self):
        """Compute Maximum A Posterior estimates."""
        # Prepare and fit MAP
        self._prepare(map_=True)

        # Write estimates to params_est.
        for param_name in self.param_names:
            self.params_est[param_name] = self.group_params[param_name].value

        return self

    def _prepare(self, retry=True):
        """Create model."""

        ############################################################
        # Try multiple times to create model. Sometimes bad initial
        # parameters are chosen randomly that yield -Inf
        # log-likelihood which causes PyMC to choke.
        model_yields_zero_prob = True
        tries = 0
        if retry:
            while (model_yields_zero_prob):
                try:
                    self._set_params()
                    model_yields_zero_prob = False
                except pm.ZeroProbability:
                    tries += 1
                    if tries > 20:
                        raise pm.ZeroProbability("Model creation failed")
        else:
            self._set_params()

        return self
    
    def _sample(self, samples=10000, burn=5000, thin=2, verbose=0, dbname=None):
        """Draw posterior samples. Requires self.model to be set.
        """
        try:
            self.mcmc_model.sample(samples, burn=burn, thin=thin, verbose=verbose)
        except NameError:
            raise NameError("mcmc_model not set, call ._prepare()")

        self._gen_stats()
        if self.is_subj_model:
            self._gen_stats_subjs()
            
        if dbname is not None:
            self.mcmc_model.db.commit()

        return self

    def norm_approx(self):
        # Set model parameter values to Normal Approximations
        self._prepare()
        
        self.norm_approx_model = pm.NormApprox(self.model)
        self.norm_approx_model.fit()

        return self
            

    def mcmc(self, samples=10000, burn=5000, thin=2, verbose=0, dbname=None, map_=True):
        """Main method for sampling. Creates and initializes the model and starts sampling.
        """
        # Set and initialize model
        self._prepare()

        # Set model parameter values to MAP estimates
        if map_:
            self.map_model = pm.MAP(self.model)
            self.map_model.fit()

        # Save future samples to database if needed.
        if dbname is None:
            self.mcmc_model = pm.MCMC(self.model, verbose=verbose)
        else:
            self.mcmc_model = pm.MCMC(self.model, db='sqlite', dbname=dbname, verbose=verbose)

        # Draw samples
        self._sample(samples=samples, burn=burn, thin=thin, verbose=verbose, dbname=dbname)

        return self
    
    def _set_traces(self, params, mcmc_model=None, add=False):
        """Externally set the traces of group_params. This is needed
        when loading a model from a previous sampling run saved to a
        database.
        """
        if not mcmc_model:
            mcmc_model = self.mcmc_model
        for param_name, param_inst in params.iteritems():
            try:
                if add:
                    param_inst.trace._trace[0] = np.concatenate((param_inst.trace._trace[0],
                                                                 mcmc_model.trace(param_name)()))
                else:
                    param_inst.trace = mcmc_model.trace(param_name)
            except AttributeError: # param_inst is an array
                if self.trace_subjs:
                    for i, subj_param in enumerate(param_inst):
                        if add:
                            subj_param.trace._trace[0] = np.concatenate((subj_param.trace._trace[0],
                                                                         mcmc_model.trace('%s%i'%(param_name,i))()))
                        else:
                            subj_param.trace = mcmc_model.trace('%s%i'%(param_name,i))

    def mcmc_load_from_db(self, dbname):
        """Load samples from a database created by an earlier model
        run (e.g. by calling .mcmc(dbname='test'))
        """
        # Set up model
        self._prepare()
        
        # Open database
        db = pm.database.sqlite.load(dbname)

        # Create mcmc instance reading from the opened database
        self.mcmc_model = pm.MCMC(self.model, db=db, verbose=verbose)

        # Take the traces from the database and feed them into our
        # distribution variables (needed for _gen_stats())
        self._set_traces(self.group_params)
        self._gen_stats()

        if self.is_subj_model:
            self._set_traces(self.group_params_tau)
            self._set_traces(self.subj_params)
            self._gen_stats_subjs()

        return self
    
    def _gen_stats(self, save_stats_to=None):
        for param_name in self.param_names:
            self.params_est[param_name] = np.mean(self.mcmc_model.trace(param_name)())
            self.params_est_std[param_name] = np.std(self.mcmc_model.trace(param_name)())

        # Save stats to output file
        if save_stats_to is not None:
            print "Saving stats to %s" % save_stats_to
            with open(save_stats_to, 'w') as fd:
                for name, value in self.params_est.iteritems():
                    fd.write('%s: %f\n'%(name, value))
                    
        return self

    def _gen_stats_subjs(self):
        raise NotImplementedError, "Model has no subject capabilites"

def hierarchical(c):
    """Class decorator."""
    def func(data, **kwargs):
        return Hierarchical(data, c, **kwargs)
    return func

class Hierarchical(Base):
    """Class that builds hierarchical bayesian models.
    Use the @hierarchical decorator for you model."""
    def __init__(self, data, ParamFactory, **kwargs):
        # Take out parameters for this class
        if kwargs.has_key('depends_on'):
            self.depends_on = kwargs['depends_on']
            del kwargs['depends_on']
        else:
            self.depends_on = {}

        if kwargs.has_key('effects_on'):
            self.effects_on = kwargs['effects_on']
            del kwargs['effects_on']
        else:
            self.effects_on = {}

        if kwargs.has_key('is_subj_model'):
            self.is_subj_model = kwargs['is_subj_model']
            del kwargs['is_subj_model']
        else:
            self.is_subj_model = 'subj_idx' in data.dtype.names

        super(Hierarchical, self).__init__()

        self._param_factory = ParamFactory(data=data, **kwargs)

        self.data = data
        
        if self.is_subj_model:
            self._subjs = np.unique(data['subj_idx'])
            self._num_subjs = self._subjs.shape[0]

        self.param_names = self._param_factory.get_param_names()
        self.group_params = {}
        self.group_params_tau = {}
        self.group_params_dep = {}
        self.subj_params = {}

    def _get_data_depend(self, get_group_params=False):
        """Returns a list of tuples with the data partition according to depends_on and the
        parameter distributions that go with them."""
        
        if self.is_subj_model and not get_group_params:
            params = copy(self.subj_params) # use subj parameters to feed into model
        else:
            params = copy(self.group_params) # use group parameters to feed into model

        depends_on = copy(self.depends_on)

        data_dep = self._get_data_depend_rec(self.data, depends_on, params, get_group_params=get_group_params)

        return data_dep
    
    def _get_data_depend_rec(self, data, depends_on, params, param_name=None, get_group_params=False):
        """Recursive function to partition data and params depending on classes (i.e. depends_on)."""
        if len(depends_on) != 0: # If depends are present
            data_params = []
            param_name = depends_on.keys()[0] # Get first param from depends_on
            col_name = depends_on.pop(param_name) # Take out param
            depend_elements = np.unique(data[col_name])
            for depend_element in depend_elements:
                data_dep = data[data[col_name] == depend_element]
                # Set the appropriate param
                if self.is_subj_model and not get_group_params:
                    params[param_name] = self.subj_params[param_name+str(depend_element)]
                else:
                    params[param_name] = self.group_params[param_name+str(depend_element)]
                # Recursive call with one less dependency and the sliced data.
                data_param = self._get_data_depend_rec(data_dep,
                                                       depends_on=copy(depends_on),
                                                       params=copy(params),
                                                       param_name=param_name+str(depend_element),
                                                       get_group_params=get_group_params)
                data_params += data_param
            return data_params
                
        else: # Data does not depend on anything (anymore)
            return [(data, params, param_name)]


    def _set_dependent_param(self, param_name):
        """Set group parameters that only depend on individual classes of data."""
        depends_on = self.depends_on[param_name]
        uniq_data_dep = np.unique(self.data[depends_on])

        self.group_params_dep[param_name] = []
        for pos, uniq_date in enumerate(uniq_data_dep):
            tag = str(uniq_date)
            param_tag = '%s%s'%(param_name, tag)
            pos_abs=(pos,len(uniq_data_dep)-1)
            self.group_params[param_tag] = self._param_factory.get_root_param(param_name,
                                                                              self.group_params,
                                                                              tag=tag,
                                                                              pos=pos_abs)
            self.group_params_dep[param_name].append(param_tag)
            
            if self.is_subj_model:
                self._set_subj_params(param_name, tag=tag, pos=pos_abs)

        return self

    def _set_params(self):
        """Set group level distributions. One distribution for each parameter."""
        for param_name in self.param_names: # Loop through param names
            if param_name in self.depends_on.keys():
                self._set_dependent_param(param_name)
            else:
                # Parameter does not depend on data
                # Set group parameter
                self.group_params[param_name] = self._param_factory.get_root_param(param_name, self.group_params, tag='')

                if self.is_subj_model:
                    self._set_subj_params(param_name)

        # Set likelihoods
        self._set_model()

        return self

    def _set_subj_params(self, param_name, tag=None, pos=None):
        if tag is None:
            tag = ''

        param_name_full = '%s%s'%(param_name,tag)
        #####################
        # Set subj parameter
        self.subj_params[param_name_full] = np.empty(self._num_subjs, dtype=object)

        # Generate subj variability parameter
        param_inst_tau = self._param_factory.get_tau_param(param_name_full, tag='tau')
        self.group_params_tau[param_name_full] = param_inst_tau
        param_inst = self.group_params[param_name_full]
                    
        for subj_idx,subj in enumerate(self._subjs):
            self.subj_params[param_name_full][subj_idx] = self._param_factory.get_subj_param(param_name,
                                                                                             param_inst,
                                                                                             param_inst_tau,
                                                                                             int(subj),
                                                                                             self.subj_params,
                                                                                             tag=tag,
                                                                                             pos=pos)


        return self
    
    def _set_model(self):
        """Create and set up the complete model."""
        data_dep = self._get_data_depend()

        self.likelihoods = []
        
        for i, (data, params, param_name) in enumerate(data_dep):
            self.likelihoods.append(self._create_model(data, params, i))
            
        # Set and return all distributions belonging.
        self.model = self.likelihoods + self.group_params.values() + self.group_params_tau.values() + self.subj_params.values()

        return self
        
    def _create_model(self, data, params, idx):
        """Create and return a model on [data] with [params].
        """
        if self.is_subj_model:
            likelihood = np.empty(self._num_subjs, dtype=object)
            for i,subj in enumerate(self._subjs):
                data_subj = data[data['subj_idx'] == subj] # Select data belonging to subj

                likelihood[i] = self._param_factory.get_model("model_%i_%i"%(idx, i), data_subj, params, idx=i)
        else: # Do not use subj params, but group ones
            likelihood = self._param_factory.get_model("model_%i"%idx, data, params)

        return likelihood

    def summary(self, delimiter=None):
        """Return summary statistics of the fit model."""
        if delimiter is None:
            delimiter = '\n'

        s = ''
        
        for param, depends_on in self.depends_on.iteritems():
            s+= 'Parameter "%s" depends on: %s%s' %(param, ','.join(depends_on), delimiter)

        s += delimiter + 'General model stats:' + delimiter
        for name, value in self.stats.iteritems():
            s += '%s: %f%s'%(name, value, delimiter) 

        s += delimiter + 'Group parameter\t\t\t\tMean\tStd\t95%' + delimiter
        for name, value in self.params_est.iteritems():
            # Create appropriate number of tabs for correct displaying
            # if parameter names are longer than one tab space.
            # 5 tabs if name string is smaller than 4 letters.
            num_tabs = int(6-np.ceil((len(name)/8.)))
            tabs = ''.join(['\t' for i in range(num_tabs)])
            s += '%s%s%.2f\t%.2f\t%.2f, %.2f%s'%(name, tabs, value,
                                        self.params_est_std[name],
                                        self.params_est_perc[name][0],
                                        self.params_est_perc[name][1],
                                        delimiter)

        return s

    def summary_subjs(self, delimiter=None):
        if delimiter is None:
            delimiter = '\n'

        s = 'Group parameter\t\t\t\tMean\t\Std' + delimiter
        for subj, params in self.params_est_subj.iteritems():
            s += 'Subject: %i%s' % (subj, delimiter)
            for name,value in params.iteritems():
                # Create appropriate number of tabs for correct displaying
                # if parameter names are longer than one tab space.
                num_tabs = 6-np.ceil((len(name)/8.))
                tabs = ''.join(['\t' for i in range(num_tabs)])
                s += '%s%s%.2f\t%.2f%s'%(name, tabs, value, self.params_est_subj_std[subj][name], delimiter)
            s += delimiter
            
        return s

    def compare_all_pairwise(self):
        for params in self.group_params_dep.itervalues():
            for p0,p1 in kabuki.utils.all_pairs(params):
                diff = self.group_params[p0].trace()-self.group_params[p1].trace()
                print "%s vs %s: %.3f" %(p0, p1, np.mean(diff))
                
    def _gen_stats(self):
        """Generate summary statistics of fit model."""
        self.stats['logp'] = self.mcmc_model.logp
        self.stats['dic'] = self.mcmc_model.dic
        
        for param_name in self.group_params.iterkeys():
            self.params_est[param_name] = np.mean(self.mcmc_model.trace(param_name)())
            self.params_est_std[param_name] = np.std(self.mcmc_model.trace(param_name)())
            sorted_trace = np.sort(self.mcmc_model.trace(param_name)())
            self.params_est_perc[param_name] = (sorted_trace[int(.05*len(sorted_trace)-1)],
                                                sorted_trace[int(.95*len(sorted_trace)-1)])
            
        return self
    
    def _gen_stats_subjs(self):
        self.params_est_subj = {}
        self.params_est_subj_std = {}
        self.params_est_subj_perc = {}

        # Initialize params_est_subj arrays
        for subj_idx in range(self._num_subjs):
            if not self.params_est_subj.has_key(subj_idx):
                self.params_est_subj[subj_idx] = {}
                self.params_est_subj_std[subj_idx] = {}

        # Generate stats
        for name,params in self.subj_params.iteritems():
            for subj_idx,subj_dist in enumerate(params):
                if subj_dist is None:
                    continue # z is none in non-bias case

                self.params_est_subj[subj_idx][name] = np.mean(subj_dist.trace())
                self.params_est_subj_std[subj_idx][name] = np.std(subj_dist.trace())

        return self

    def save_stats(self, fname):
        """Save stats to output file."""
        print "Saving stats to %s" % fname
        s = self.summary()
        with open(fname, 'w') as fd:
            fd.write(s)
                
        return self


#@hierarchical
class Prototype(object):
    def __init__(self, data):
        self.data = data

    def get_param_names(self):
        return ('param1', 'param2')
    
    def get_root_param(self, param, tag=None):
        return None

    def get_tau_param(self, param, tag=None):
        return None
    
    def get_subj_param(self, param_name, parent_mean, parent_tau, subj_idx):
        return pm.Normal('%s%i'%(param_name, subj_idx), mu=parent_mean, tau=parent_tau)

@hierarchical
class MANOVA(Prototype):
    def get_param_names(self):
        return ('score',)

    
@hierarchical
class Regression(Prototype):
    def get_param_names(self):
        return ('theta', 'x')
    
    def get_root_param(self, param, tag=None):
        return pm.Uniform('%s%s'%(param,tag), lower=0, upper=50)

    def get_tau_param(self, param, tag=None):
        return None
    
    def get_model(self, name, subj_data, params, idx=None):
        @deterministic(plot=False)
        def modelled_y(x=params['x'], theta=params['theta']):
            """Return y computed from the straight line model, given the
            inferred true inputs and the model paramters."""
            slope, intercept = theta
            return slope*x + intercept

        return (pm.Normal(name, mu=modelled_y, tau=2, value=subj_data['dependent'], observed=True), modelled_y)

@hierarchical        
class Effect(Prototype):
    def get_param_names(self):
        return ('base', 'effect')
    
    def get_root_param(self, param, all_params, tag=None, pos=None):
        if pos is not None:
            # Check if last element
            if pos[0] == pos[1]:
                param_full_name = '%s%s'%(param,tag)
                # Set parameter so that they sum to zero
                args = tuple([all_params[p] for p in all_params if p.startswith(param)])
                return pm.Deterministic(kabuki.utils.neg_sum,
                                        param_full_name,
                                        param_full_name,
                                        parents={'args':args})

        return pm.Uniform('%s%s'%(param,tag), lower=-5, upper=5)

    def get_tau_param(self, param, tag=None):
        return pm.Uniform('%s%s'%(param,tag), lower=0, upper=800, plot=False)
    
    def get_subj_param(self, param_name, parent_mean, parent_tau, subj_idx, all_params, tag=None, pos=None):
        param_full_name = '%s%s%i'%(param_name,tag,subj_idx)
        if pos is not None:
            if pos[0] == pos[1]:
                # Set parameter so that they sum to zero
                print all_params
                args = tuple([all_params[p][subj_idx] for p in all_params if p.startswith(param_name) and all_params[p][subj_idx] is not None])
                return pm.Deterministic(kabuki.utils.neg_sum,
                                        param_full_name,
                                        param_full_name,
                                        parents={'args':args})
                                        
        return pm.Normal(param_full_name, mu=parent_mean, tau=parent_tau, plot=False)

    def get_model(self, name, subj_data, params, idx=None):
        return pm.Normal(name, value=subj_data['score'], mu=params['base'][idx]+params['effect'][idx], tau=2,
                         observed=True, plot=False)
    

