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
    """Base class for the hierarchical bayesian drift diffusion
    model."""
    def __init__(self):
        self.model = None
        self.mcmc_model = None
        self.map_model = None
        self.params_est = {}
        self.params_est_std = {}
        self.stats = {}
	self.colors = ('r','b','g','y','c')

    def _set_group_params(self):
        raise NotImplementedError("This method has to be overloaded")
    
    def _set_model(self):
        raise NotImplementedError("This method has to be overloaded")

    def _set_all_params(self):
        self._set_group_params()


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
                    self._set_all_params()
                    model_yields_zero_prob = False
                except pm.ZeroProbability:
                    tries += 1
                    if tries > 20:
                        raise pm.ZeroProbability("Model creation failed")
        else:
            self._set_all_params()

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
                                                                         mcmc_model.trace('%s_%i'%(param_name,i))()))
                        else:
                            subj_param.trace = mcmc_model.trace('%s_%i'%(param_name,i))

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

        if kwargs.has_key('is_subj_model'):
            self.is_subj_model = kwargs['is_subj_model']
            del kwargs['is_subj_model']
        else:
            self.is_subj_model = 'subj_idx' in data.dtype.names

        super(Hierarchical, self).__init__()

        self.param_factory = ParamFactory(data=data, **kwargs)

        self.data = data
        
        if self.is_subj_model:
            self._subjs = np.unique(data['subj_idx'])
            self._num_subjs = self._subjs.shape[0]

        self.param_names = self.param_factory.get_param_names()
        self.group_params = {}
        self.subj_params = {}
        self.root_params = {}
        self.group_params_tau = {}
        self.root_params_tau = {}

    def _set_all_params(self):
        self._set_group_params()
        if self.is_subj_model:
            self._set_subj_params()

        self._set_model()

        return self

    def _set_dependent_group_param(self, param):
        """Set group parameters that only depend on individual classes of data."""
        depends_on = self.depends_on[param]
        uniq_data_dep = np.unique(self.data[depends_on])

        for uniq_date in uniq_data_dep:
            tag = str(uniq_date)
            param_tag = '%s_%s'%(param, tag)
            self.group_params[param_tag] = self.param_factory.get_root_param(param, tag=tag)

        return self

    def _set_group_params(self):
        """Set group level distributions. One distribution for each parameter."""
        for param in self.param_names: # Loop through param names
            if param in self.depends_on.keys():
                self._set_dependent_group_param(param)
            else:
                # Parameter does not depend on data
                self.group_params[param] = self.param_factory.get_root_param(param)
        
        return self

    def _set_subj_params(self):
        """Set individual subject distributions. Each subject is
        assigned one set of parameter distributions which have the
        group level parameters as their parents"""
        # For each global param, create n subj_params
        self.subj_params = {}

        # Initialize
        for param_name, param_inst in self.group_params.iteritems():
            self.subj_params[param_name] = np.empty(self._num_subjs, dtype=object)
            
        for param_name, param_inst in self.group_params.iteritems():
            # Create tau parameter for global param
            param_inst_tau = self.param_factory.get_tau_param(param_name)
            self.group_params_tau[param_name] = param_inst_tau
            # Create num_subjs individual subject parameters
            for subj_idx,subj in enumerate(self._subjs):
                self.subj_params[param_name][subj_idx] = self.param_factory.get_subj_param(param_name,
                                                                                           param_inst,
                                                                                           param_inst_tau,
                                                                                           int(subj))
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
        

    def _get_data_depend(self, get_group_params=False):
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
                    params[param_name] = self.subj_params[param_name+'_'+str(depend_element)]
                else:
                    params[param_name] = self.group_params[param_name+'_'+str(depend_element)]
                # Recursive call with one less dependency and the sliced data.
                data_param = self._get_data_depend_rec(data_dep,
                                                       depends_on=copy(depends_on),
                                                       params=copy(params),
                                                       param_name=param_name+'_'+str(depend_element),
                                                       get_group_params=get_group_params)
                data_params += data_param
            return data_params
                
        else: # Data does not depend on anything (anymore)
            return [(data, params, param_name)]


    def _create_model(self, data, params, idx):
        """Create and return a model on [data] with [params].
        """
        if self.is_subj_model:
            likelihood = np.empty(self._num_subjs, dtype=object)
            for i,subj in enumerate(self._subjs):
                data_subj = data[data['subj_idx'] == subj] # Select data belonging to subj

                likelihood[i] = self.param_factory.get_model("model_%i_%i"%(idx, i), data_subj, params, idx=i)
        else: # Do not use subj params, but group ones
            likelihood = self.param_factory.get_model("model_%i"%idx, data, params)

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

        s += delimiter + 'Group parameter\t\t\t\tMean\t\tStd' + delimiter
        for name, value in self.params_est.iteritems():
            # Create appropriate number of tabs for correct displaying
            # if parameter names are longer than one tab space.
            # 5 tabs if name string is smaller than 4 letters.
            num_tabs = int(6-np.ceil((len(name)/8.)))
            tabs = ''.join(['\t' for i in range(num_tabs)])
            s += '%s%s%f\t%f%s'%(name, tabs, value, self.params_est_std[name], delimiter)

        return s

    def summary_subjs(self, delimiter=None):
        if delimiter is None:
            delimiter = '\n'

        s = 'Group parameter\t\t\t\tMean\t\tStd' + delimiter
        for subj, params in self.params_est_subj.iteritems():
            s += 'Subject: %i%s' % (subj, delimiter)
            for name,value in params.iteritems():
                # Create appropriate number of tabs for correct displaying
                # if parameter names are longer than one tab space.
                num_tabs = 5-np.ceil((len(name)/4.))
                tabs = ''.join(['\t' for i in range(num_tabs)])
                s += '%s%s%f\t%f%s'%(name, tabs, value, self.params_est_subj_std[subj][name], delimiter)
            s += delimiter
            
        return s
    
    def _gen_stats(self):
        """Generate summary statistics of fit model."""
        self.stats['logp'] = self.mcmc_model.logp
        self.stats['dic'] = self.mcmc_model.dic

        self.params_est_subj = {}
        self.params_est_subj_std = {}
        
        for param_name in self.group_params.iterkeys():
            self.params_est[param_name] = np.mean(self.mcmc_model.trace(param_name)())
            self.params_est_std[param_name] = np.std(self.mcmc_model.trace(param_name)())

        for param_name in self.root_params.iterkeys():
            self.params_est[param_name] = np.mean(self.mcmc_model.trace(param_name)())
            self.params_est_std[param_name] = np.std(self.mcmc_model.trace(param_name)())

        return self
    
    def _gen_stats_subjs(self):
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
