 #!/usr/bin/python
from __future__ import division
from copy import copy

import numpy as np
import numpy.lib.recfunctions as rec
from ordereddict import OrderedDict

import pymc as pm

import kabuki

class Hierarchical(object):
    """Class that builds hierarchical bayesian models.

    This class can best be used with the @hierarchical decorator
    applied to a user-defined class providing parameter creation
    functions (e.g. see models.ANOVA)."""
    def __init__(self, data, is_group_model=None, depends_on=None):
        """Initialize hierarchical model.

        Arguments:
        ==========

        data <numpy.recarray>: Structured array containing input data.

        Keyword arguments:
        ==================

        is_group_model <bool>: Model is a subject model. This results
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
        self.params_est = {}
        self.params_est_std = {}
        self.params_est_perc = {}
        self.stats = {}

        self.data = data

        if depends_on is None:
            self.depends_on = {}
        else:
            self.depends_on = depends_on

        if is_group_model is None:
            self.is_group_model = 'subj_idx' in data.dtype.names
        else:
            self.is_group_model = is_group_model

        # Should the model incorporate multiple subjects
        if self.is_group_model:
            self._subjs = np.unique(data['subj_idx'])
            self._num_subjs = self._subjs.shape[0]

        self.root_nodes = OrderedDict()
        self.root_nodes_tau = OrderedDict()
        self.root_nodes_dep = OrderedDict()
        self.child_nodes = OrderedDict()

    def _get_data_depend(self, get_group_params=False):
        """Partition data according to self.depends_on.

        Returns:
        ========
        
        List of tuples with the data, the corresponding parameter
        distribution and the parameter name."""
        
        if self.is_group_model and not get_group_params:
            params = copy(self.child_nodes) # use subj parameters to feed into model
        else:
            params = copy(self.root_nodes) # use group parameters to feed into model

        depends_on = copy(self.depends_on)

        # Make call to recursive function that does the partitioning
        data_dep = self._get_data_depend_rec(self.data, depends_on, params, '', get_group_params=get_group_params)

        return data_dep
    
    def _get_data_depend_rec(self, data, depends_on, params, dep_name, param_name=None, get_group_params=False):
        """Recursive function to partition data and params according
        to depends_on."""
        if len(depends_on) != 0: # If depends are present
            data_params = []
            # Get first param from depends_on
            param_name = depends_on.keys()[0]
            col_name = depends_on.pop(param_name) # Take out param
            depend_elements = np.unique(data[col_name])
            # Loop through unique elements
            for depend_element in depend_elements:
                # Extract rows containing unique element
                data_dep = data[data[col_name] == depend_element]
                # Add a key that is only the col_name that links to
                # the correct dependent nodes. This is the central
                # trick so that later on the get_rootless_child can use
                # params[col_name] and the observed will get linked to
                # the correct nodes automatically.
                if self.is_group_model and not get_group_params:
                    params[param_name] = self.child_nodes[param_name+str(depend_element)]
                else:
                    params[param_name] = self.root_nodes[param_name+str(depend_element)]
                # Recursive call with one less dependency and the selected data.
                data_param = self._get_data_depend_rec(data_dep,
                                                       depends_on=copy(depends_on),
                                                       params=copy(params),
                                                       dep_name = str(depend_element),
                                                       param_name=param_name,
                                                       get_group_params=get_group_params)
                data_params += data_param
            return data_params
                
        else: # Data does not depend on anything (anymore)
            return [(data, params, param_name, dep_name)]

    def create(self):
        """Set group level distributions. One distribution for each
        parameter."""
        for param_name, has_root in self.param_names: # Loop through param names
            if has_root:
                # Check if parameter depends on data
                if param_name in self.depends_on.keys():
                    self._set_dependent_param(param_name)
                else:
                    self._set_independet_param(param_name)
            else:
                self._set_rootless_child_nodes(param_name)

        # Create model dictionary
        nodes = {}
        for name,node in self.root_nodes.iteritems():
            nodes[name+'_root'] = node
        for name,node in self.child_nodes.iteritems():
            nodes[name+'_childs'] = node
        for name,node in self.root_nodes_tau.iteritems():
            nodes[name+'_tau'] = node

        return nodes
    
    def _set_dependent_param(self, param_name):
        """Set parameter that depends on data.

        Arguments:
        ==========
        
        param_name<string>: Name of parameter that depends on data for
        which to set distributions."""

        # Get column names for provided param_name
        depends_on = self.depends_on[param_name]
        # Get unique elements from the columns
        data_dep = self.data[depends_on]
        uniq_data_dep = np.unique(data_dep)
        self.root_nodes_dep[param_name] = []

        # Loop through unique elements
        for uniq_date in uniq_data_dep:
            # Select data
            data_dep_select = self.data[(data_dep == uniq_date)]

            # Create name for parameter
            tag = str(uniq_date)
            param_tag = '%s%s'%(param_name, tag)

            # Create parameter distribution from factory
            self.root_nodes[param_tag] = self.get_root_node(param_name, self.root_nodes, tag, data_dep_select)
            self.root_nodes_dep[param_name].append(param_tag)
            
            if self.is_group_model:
                # Create appropriate subj parameter
                self._set_child_nodes(param_name, tag, data_dep_select)

        return self

    def _set_independet_param(self, param_name):
        # Parameter does not depend on data
        # Set group parameter
        self.root_nodes[param_name] = self.get_root_node(param_name, self.root_nodes, '', self.data)

        if self.is_group_model:
            self._set_child_nodes(param_name, '', self.data)
        
        return self

    def _set_child_nodes(self, param_name, tag, data):
        param_name_full = '%s%s' % (param_name, tag)
        # Generate subj variability parameter tau
        param_inst_tau = self.get_tau_node(param_name_full, self.root_nodes_tau, 'tau')
        self.root_nodes_tau[param_name_full] = param_inst_tau

        param_inst = self.root_nodes[param_name_full]

        # Init
        self.child_nodes[param_name_full] = np.empty(self._num_subjs, dtype=object)
        # Create subj parameter distribution for each subject
        for subj_idx,subj in enumerate(self._subjs):
            data_subj = data[data['subj_idx']==subj_idx]
            self.child_nodes[param_name_full][subj_idx] = self.get_child_node(param_name, param_inst, param_inst_tau, 
                                                                              subj_idx, self.child_nodes, tag, data_subj)
        return self
    
    def _set_rootless_child_nodes(self, param_name):
        """Create and set up the complete model."""
        # Divide data and parameter distributions according to self.depends_on
        data_dep = self._get_data_depend()
        # Loop through parceled data and params and create an observed stochastic
        for i, (data, params_dep, param_dep_name, dep_name) in enumerate(data_dep):
            kabuki.debug_here()
            if param_dep_name is None:
                param_dep_name = ''
            self.child_nodes[param_name+dep_name] = self._create_rootless_child_node(param_name, data, params_dep, param_dep_name, dep_name, i)
            
        return self
        
    def _create_rootless_child_node(self, param_name, data, params, child_depends_on, dep_name, idx):
        """Create and return observed distribution where data depends
        on params.
        """
        if self.is_group_model:
            # Create observed stochastic for each subject
            rootless_child_node = np.empty(self._num_subjs, dtype=object)
            for i,subj in enumerate(self._subjs):
                # Select data belonging to subj
                data_subj = data[data['subj_idx'] == subj]
                # Select params belonging to subject
                selected_child_nodes = {}
                # Create new params dict and copy over nodes
                for name, nodes in self.child_nodes.iteritems():
                    selected_child_nodes[name] = nodes[i]
                if child_depends_on != '':
                    selected_child_nodes[child_depends_on] = params[child_depends_on][i] # We have to overwrite the dependent one separately
                # Call to the user-defined function!
                rootless_child_node[i] = self.get_rootless_child(param_name, "%s%i"%(dep_name, idx), data_subj, selected_child_nodes, idx=i)
        else: # Do not use subj params, but group ones
            rootless_child_node = self.get_rootless_child(param_name, "%s"%dep_name, data, params)

        return rootless_child_node

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
            value = self.params_est[name]
            num_tabs = int(5-np.floor(((len(name))/8.)))
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
                num_tabs = int(5-np.floor(((len(name))/8.)))
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
        for params in self.root_nodes_dep.itervalues():
            # Loop through all pairwise combinations
            for p0,p1 in kabuki.utils.all_pairs(params):
                diff = self.root_nodes[p0].trace()-self.root_nodes[p1].trace()
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
        for i,(p0,p1) in enumerate(kabuki.utils.all_pairs(self.root_nodes.values())):
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
        
        for param_name in self.root_nodes.iterkeys():
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
        for name,params in self.child_nodes.iteritems():
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
