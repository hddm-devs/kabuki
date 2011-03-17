#!/usr/bin/python
from __future__ import division
from copy import copy

import numpy as np
import numpy.lib.recfunctions as rec
from ordereddict import OrderedDict

import pymc as pm

import kabuki


# Model classes
class Base(object):
    """Base class for MCMC models."""
    def __init__(self, *args, **kwargs):
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
                except pm.ZeroProbability as e:
                    self.group_params = OrderedDict()
                    self.group_params_tau = OrderedDict()
                    self.group_params_dep = OrderedDict()
                    self.subj_params = OrderedDict()
                    tries += 1
                    if tries > 20:
                        print e
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
            

    def mcmc(self, samples=10000, burn=5000, thin=2, verbose=0, step_method=None, dbname=None, map_=True, retry=True, sample=True):
        """Main method for sampling. Creates and initializes the model
        and starts sampling.
        """
        # Set and initialize model
        self._prepare(retry=retry)

        # Set model parameter values to MAP estimates
        if map_:
            self.map_model = pm.MAP(self.model)
            self.map_model.fit()

        # Create MCMC instance
        if dbname is None:
            self.mcmc_model = pm.MCMC(self.model, verbose=verbose)
        else:
            # Save future samples to database if needed.
            self.mcmc_model = pm.MCMC(self.model, db='sqlite', dbname=dbname, verbose=verbose)

        if step_method is not None:
            self.mcmc_model.use_step_method(step_method, self.group_params.values() + self.group_params_tau.values() + [j for i in self.subj_params.values() for j in i])
            
        #self.mcmc_model.use_step_method(pm.Gibbs, self.group_params.values())
        # Start sampling
        if sample:
            self._sample(samples=samples, burn=burn, thin=thin, verbose=verbose, dbname=dbname)

        return self
    
    def _set_traces(self, params, mcmc_model=None, add=False):
        """Externally set the traces of group_params. This is needed
        when loading a model from a previous sampling run saved to a
        database.
        """
        if not mcmc_model:
            mcmc_model = self.mcmc_model

        # Loop through parameters and set traces
        for param_name, param_inst in params.iteritems():
            try:
                if add:
                    # Append trace
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
        """Generate mean and std statistics of each distributions and
        save to params_est.

        Arguments:
        ==========
        
        save_stats_to <string>: filename to save stats to."""
        # Loop through params and generate statistics
        for param_name in self.param_names:
            # Mean
            self.params_est[param_name] = np.mean(self.mcmc_model.trace(param_name)())
            # Std
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
    """Hierarchical decorator for classes.

    Use this decorator for your class that returns
    the model parameter distributions and likelihoods.

    E.g.:

    @hierarchical
    class MyClass(object):
        # Class definition
        ...
    """
    # This little trick with returning a new Class is so
    # that decorated classes can do inheritance and call
    # super(). The pattern was suggested by lrh9 in #python
    # on irc.freenode.net
    class Hierarchical(HierarchicalBase):
        def __init__(self, data, **kwargs):
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

            # Call parent's __init__
            super(self.__class__, self).__init__(data, **kwargs)

            # Link to the decorated object
            self._param_factory = c(data, **kwargs)
            
            self.param_names = self._param_factory.param_names
            
    return Hierarchical

class HierarchicalBase(Base):
    """Class that builds hierarchical bayesian models.

    This class can best be used with the @hierarchical decorator
    applied to a user-defined class providing parameter creation
    functions (e.g. see models.ANOVA)."""
    def __init__(self, data, **kwargs):
        """Initialize hierarchical model.

        Arguments:
        ==========

        data <numpy.recarray>: Structured array containing input data.

        Keyword arguments:
        ==================

        is_subj_model <bool>: Model is a subject model. This results
        in a hierarchical model with distributions for each parameter
        for each subject whose parameters are themselves distributed
        according to a group parameter distribution.
        
        depends_on <dict>: Specifies which parameter depends on data
        in a supplied column. For each unique element in that column,
        a separate set of parameter distributions will be created and
        applied. Multiple columns can be specified in a sequential
        container (e.g. list)

        Example: depends_on={'param1':['column1']}
        
        Suppose column1 has the elements 'element1' and 'element2',
        then parameters 'param1('element1',)' and
        'param1('element2',)' will be created and the corresponding
        parameter distribution and data will be provided to the
        user-specified method get_liklihood().
        """
        # Call parent's class __init__
        super(HierarchicalBase, self).__init__()

        # Should the model incorporate multiple subjects
        if self.is_subj_model:
            self._subjs = np.unique(data['subj_idx'])
            self._num_subjs = self._subjs.shape[0]

        self.group_params = OrderedDict()
        self.group_params_tau = OrderedDict()
        self.group_params_dep = OrderedDict()
        self.subj_params = OrderedDict()

    def __getattr__(self, name):
        if name in dir(self._param_factory):
            return self._param_factory.__getattribute__(name)
        else:
            raise AttributeError, "hierarchical nor param_factory object have attribute '%s'" % name

    def _get_data_depend(self, get_group_params=False):
        """Partition data according to self.depends_on.

        Returns:
        ========
        
        List of tuples with the data, the corresponding parameter
        distribution and the parameter name."""
        
        if self.is_subj_model and not get_group_params:
            params = copy(self.subj_params) # use subj parameters to feed into model
        else:
            params = copy(self.group_params) # use group parameters to feed into model

        depends_on = copy(self.depends_on)

        # Make call to recursive function that does the partitioning
        data_dep = self._get_data_depend_rec(self._param_factory.data, depends_on, params, get_group_params=get_group_params)

        return data_dep
    
    def _get_data_depend_rec(self, data, depends_on, params, param_name=None, get_group_params=False):
        """Recursive function to partition data and params according
        to depends_on."""
        if len(depends_on) != 0: # If depends are present
            data_params = []
            param_name = depends_on.keys()[0] # Get first param from depends_on
            col_name = depends_on.pop(param_name) # Take out param
            depend_elements = np.unique(data[col_name])
            # Loop through unique elements
            for depend_element in depend_elements:
                # Extract rows containing unique element
                data_dep = data[data[col_name] == depend_element]
                # Set the appropriate parameter
                if self.is_subj_model and not get_group_params:
                    params[param_name] = self.subj_params[param_name+str(depend_element)]
                else:
                    params[param_name] = self.group_params[param_name+str(depend_element)]
                # Recursive call with one less dependency and the selected data.
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
        """Set parameter that depends on data.

        Arguments:
        ==========
        
        param_name<string>: Name of parameter that depends on data for
        which to set distributions."""

        # Get column names for provided param_name
        depends_on = self.depends_on[param_name]
        # Get unique elements from the columns
        uniq_data_dep = np.unique(self._param_factory.data[depends_on])

        self.group_params_dep[param_name] = []

        # Loop through unique elements
        for pos, uniq_date in enumerate(uniq_data_dep):
            # Create name for parameter
            tag = str(uniq_date)
            param_tag = '%s%s'%(param_name, tag)
            # The how maniest element is it
            pos_abs=(pos,len(uniq_data_dep)-1)

            # Create parameter distribution from factory
            self.group_params[param_tag] = self._param_factory.get_root_param(param_name,
                                                                              self.group_params,
                                                                              tag,
                                                                              pos=pos_abs)
            self.group_params_dep[param_name].append(param_tag)
            
            if self.is_subj_model:
                # Create appropriate subj parameter
                self._set_subj_params(param_name, tag, pos)

        return self

    def _set_params(self):
        """Set group level distributions. One distribution for each
        parameter."""
        for param_name in self.param_names: # Loop through param names
            # Check if parameter depends on data
            if param_name in self.depends_on.keys():
                self._set_dependent_param(param_name)
            else:
                # Parameter does not depend on data
                # Set group parameter
                self.group_params[param_name] = self._param_factory.get_root_param(param_name, self.group_params, '')

                if self.is_subj_model:
                    self._set_subj_params(param_name, '')

        # Set likelihoods
        self._set_model()

        return self

    def _set_subj_params(self, param_name, tag, pos=None):
        param_name_full = '%s%s' % (param_name, tag)

        # Init
        self.subj_params[param_name_full] = np.empty(self._num_subjs, dtype=object)

        # Generate subj variability parameter tau
        param_inst_tau = self._param_factory.get_tau_param(param_name_full, self.group_params_tau, 'tau')
        self.group_params_tau[param_name_full] = param_inst_tau

        param_inst = self.group_params[param_name_full]

        # Create subj parameter distribution for each subject
        for subj_idx,subj in enumerate(self._subjs):
            self.subj_params[param_name_full][subj_idx] = self._param_factory.get_subj_param(param_name,
                                                                                             param_inst,
                                                                                             param_inst_tau,
                                                                                             int(subj),
                                                                                             self.subj_params,
                                                                                             tag,
                                                                                             pos=pos)
        return self
    
    def _set_model(self):
        """Create and set up the complete model."""
        # Divide data and parameter distributions according to self.depends_on
        data_dep = self._get_data_depend()

        self.likelihoods = []
        # Loop through parceled data and params and create an observed stochastic
        for i, (data, params, param_name) in enumerate(data_dep):
            self.likelihoods.append(self._create_observed(data, params, i))
            
        # Create list with the full model distributions, likelihoods and data
        self.model = self.likelihoods + self.group_params.values() + self.group_params_tau.values() + self.subj_params.values()

        return self
        
    def _create_observed(self, data, params, idx):
        """Create and return observed distribution where data depends
        on params.
        """
        if self.is_subj_model:
            # Create observed stochastic for each subject
            observed = np.empty(self._num_subjs, dtype=object)
            for i,subj in enumerate(self._subjs):
                data_subj = data[data['subj_idx'] == subj] # Select data belonging to subj
                # Call to the user-defined param_factory!
                observed[i] = self._param_factory.get_observed("observed_%i_%i"%(idx, i), data_subj, params, idx=i)
        else: # Do not use subj params, but group ones
            observed = self._param_factory.get_observed("observed_%i"%idx, data, params)

        return observed

    def summary(self, delimiter=None):
        """Return summary statistics of the group parameter distributions."""
        if delimiter is None:
            delimiter = '\n'

        s = ''
        
        for param, depends_on in self.depends_on.iteritems():
            s+= 'Parameter "%s" depends on: %s%s' %(param, ','.join(depends_on), delimiter)

        s += delimiter + 'General model stats:' + delimiter
        for name, value in self.stats.iteritems():
            s += '%s: %f%s'%(name, value, delimiter) 

        s += delimiter + 'Group parameter\t\t\t\tMean\tStd\t5%\t95%' + delimiter
        # Sort param names for better display
        param_names = np.sort(self.params_est.keys())
        for name in param_names:
            # Create appropriate number of tabs for correct displaying
            # if parameter names are longer than one tab space.
            # 5 tabs if name string is smaller than 8 letters.
            # TODO: Bugfix offsetting
            value = self.params_est[name]
            num_tabs = int(6-np.floor(((len(name))/7.)))
            print num_tabs
            tabs = ''.join(['\t' for i in range(num_tabs)])
            s += '%s%s%.3f\t%.3f\t%.3f\t%.3f%s'%(name, tabs, value,
                                        self.params_est_std[name],
                                        self.params_est_perc[name][0],
                                        self.params_est_perc[name][1],
                                        delimiter)

        return s

    def summary_subjs(self, delimiter=None):
        """Return summary statistics of the subject parameter distributions."""
        if delimiter is None:
            delimiter = '\n'

        s = 'Group parameter\t\t\t\tMean\t\Std' + delimiter
        for subj, params in self.params_est_subj.iteritems():
            s += 'Subject: %i%s' % (subj, delimiter)
            # Sort param names for better display
            param_names = np.sort(params.keys())
            for name in param_names:
                # Create appropriate number of tabs for correct displaying
                # if parameter names are longer than one tab space.
                value = params[name]
                num_tabs = int(6-np.floor(((len(name))/7.)))
                tabs = ''.join(['\t' for i in range(num_tabs)])
                s += '%s%s%.3f\t%.3f%s'%(name, tabs, value, self.params_est_subj_std[subj][name], delimiter)
            s += delimiter
            
        return s

    def compare_all_pairwise(self):
        """Perform all pairwise comparisons of dependent parameter
        distributions (as indicated by depends_on).

        Stats generated:
        ================

        * Mean difference
        * 5th and 95th percentile
        """
        
        print "Parameters\tMean difference\t5%\t95%"
        # Loop through dependent parameters and generate stats
        for params in self.group_params_dep.itervalues():
            # Loop through all pairwise combinations
            for p0,p1 in kabuki.utils.all_pairs(params):
                diff = self.group_params[p0].trace()-self.group_params[p1].trace()
                perc = kabuki.utils.percentile(diff)
                print "%s vs %s\t%.3f\t%.3f\t%.3f" %(p0, p1, np.mean(diff), perc[0], perc[1])

    def plot_all_pairwise(self):
        """Plot all pairwise posteriors to find correlations.
        """
        import matplotlib.pyplot as plt
        import scipy as sp
        import scipy.stats
        #size = int(np.ceil(np.sqrt(len(data_deps))))
        fig = plt.figure()
        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        # Loop through all pairwise combinations
        for i,(p0,p1) in enumerate(kabuki.utils.all_pairs(self.group_params.values())):
            fig.add_subplot(6,6,i+1)
            plt.plot(p0.trace(), p1.trace(), '.')
            (a_s,b_s,r,tt,stderr) = sp.stats.linregress(p0.trace(), p1.trace())
            reg = sp.polyval((a_s, b_s), (np.min(p0.trace()), np.max(p0.trace())))
            plt.plot((np.min(p0.trace()), np.max(p0.trace())), reg, '-')
            plt.xlabel(p0.__name__)
            plt.ylabel(p1.__name__)
            
            #plt.plot
            
                
    def _gen_stats(self):
        """Generate and set summary statistics of model and group parameter distributions."""
        self.stats['logp'] = self.mcmc_model.logp
        try:
            self.stats['dic'] = self.mcmc_model.dic
        except:
            self.stats['dic'] = 0.
        
        for param_name in self.group_params.iterkeys():
            self.params_est[param_name] = np.mean(self.mcmc_model.trace(param_name)())
            self.params_est_std[param_name] = np.std(self.mcmc_model.trace(param_name)())
            self.params_est_perc[param_name] = kabuki.utils.percentile(self.mcmc_model.trace(param_name)())
            
        return self
    
    def _gen_stats_subjs(self):
        """Generate and set summary statistics of subject parameter distributions."""
        self.params_est_subj = {}
        self.params_est_subj_std = {}
        self.params_est_subj_perc = {}

        # Initialize params_est_subj dicts
        for subj_idx in range(self._num_subjs):
            if not self.params_est_subj.has_key(subj_idx):
                self.params_est_subj[subj_idx] = {}
                self.params_est_subj_std[subj_idx] = {}
                self.params_est_subj_perc[subj_idx] = {}
                
        # Generate stats
        for name,params in self.subj_params.iteritems():
            for subj_idx,subj_dist in enumerate(params):
                self.params_est_subj[subj_idx][name] = np.mean(subj_dist.trace())
                self.params_est_subj_std[subj_idx][name] = np.std(subj_dist.trace())
                self.params_est_subj_perc[subj_idx][name] = kabuki.utils.percentile(subj_dist.trace())
                
        return self

    def save_stats(self, fname):
        """Save stats to output file.

        Arguments:
        ==========

        fname<string>: Filename to save stats to."""
        
        print "Saving stats to %s" % fname

        # Get summary string
        s = self.summary()

        # Write summary to file fname
        with open(fname, 'w') as fd:
            fd.write(s)
                
        return self
