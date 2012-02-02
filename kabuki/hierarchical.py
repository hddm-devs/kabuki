 #!/usr/bin/python
from __future__ import division
from copy import copy

import numpy as np
import numpy.lib.recfunctions as rec

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

import pymc as pm
import warnings

import kabuki
from copy import copy, deepcopy
from matplotlib.mlab import rec_drop_fields


class Parameter(object):
    """Specify a parameter of a model.

    :Arguments:
        name <str>: Name of parameter.

    :Optional:
        create_group_node <bool=True>: Create group nodes for parameter.
        create_subj_nodes <bool=True>: Create subj nodes for parameter.
        is_bottom_node <bool=False>: Is node at the bottom of the hierarchy (e.g. likelihoods)
        lower <float>: Lower bound (e.g. for a uniform distribution).
        upper <float>: Upper bound (e.g. for a uniform distribution).
        init <float>: Initialize to value.
        vars <dict>: User-defined variables, can be anything you later
            want to access.
        optional <bool=False>: Only create distribution when included.
            Otherwise, set to default value (see below).
        default <float>: Default value if optional=True.
        verbose <int=0>: Verbosity.
        var_type <string>: type of the var node, can be one of ['std', 'precision', 'sample_size']
    """

    def __init__(self, name, create_group_node=True, create_subj_nodes=True,
                 is_bottom_node=False, lower=None, upper=None, init=None,
                 vars=None, default=None, optional=False, var_lower=1e-3,
                 var_upper=10, var_type='std', verbose=0):
        self.name = name
        self.create_group_node = create_group_node
        self.create_subj_nodes = create_subj_nodes
        self.is_bottom_node = is_bottom_node
        self.lower = lower
        self.upper = upper
        self.init = init
        self.vars = vars
        self.optional = optional
        self.default = default
        self.verbose = verbose
        self.var_lower = var_lower
        self.var_upper = var_upper
        self.var_type = var_type

        if self.optional and self.default is None:
            raise ValueError("Optional parameters have to have a default value.")

        if self.is_bottom_node:
            self.create_group_node = False
            self.create_subj_nodes = True

        self.group_nodes = OrderedDict()
        self.var_nodes = OrderedDict()
        self.subj_nodes = OrderedDict()
        self.bottom_nodes = OrderedDict()

        # Pointers that get overwritten
        self.group = None
        self.subj = None
        self.tag = None
        self.data = None
        self.idx = None

    def reset(self):
        self.group = None
        self.subj = None
        self.tag = None
        self.data = None
        self.idx = None

    def get_full_name(self):
        if self.idx is not None:
            if self.tag is not None:
                return '%s%s%i'%(self.name, self.tag, self.idx)
            else:
                return '%s%i'%(self.name, self.idx)
        else:
            if self.tag is not None:
                return '%s%s'%(self.name, self.tag)
            else:
                return self.name

    full_name = property(get_full_name)

    def __repr__(self):
        return object.__repr__(self).replace(' object ', " '%s' "%self.name)


class Hierarchical(object):
    """Creation of hierarchical Bayesian models in which each subject
    has a set of parameters that are constrained by a group distribution.

    :Arguments:
        data : numpy.recarray
            Input data with a row for each trial.
            Must contain the following columns:
              * 'rt': Reaction time of trial in seconds.
              * 'response': Binary response (e.g. 0->error, 1->correct)
            May contain:
              * 'subj_idx': A unique ID (int) of the subject.
              * Other user-defined columns that can be used in depends_on
                keyword.

    :Optional:
        include : tuple
            If the model has optional arguments, they
            can be included as a tuple of strings here.

        is_group_model : bool
            If True, this results in a hierarchical
            model with separate parameter distributions for each
            subject. The subject parameter distributions are
            themselves distributed according to a group parameter
            distribution.

        depends_on : dict
            Specifies which parameter depends on data
            of a column in data. For each unique element in that
            column, a separate set of parameter distributions will be
            created and applied. Multiple columns can be specified in
            a sequential container (e.g. list)

            :Example:

            >>> depends_on={'param1':['column1']}

            Suppose column1 has the elements 'element1' and
            'element2', then parameters 'param1('element1',)' and
            'param1('element2',)' will be created and the
            corresponding parameter distribution and data will be
            provided to the user-specified method get_liklihood().

        trace_subjs : bool
             Save trace for subjs (needed for many
             statistics so probably a good idea.)

        plot_var : bool
             Plot group variability parameters
             (i.e. variance of Normal distribution.)

        replace_params : list of Parameters
            User defined parameters to replace the default ones.

    :Note:
        This class must be inherited. The child class must provide
        the following functions:
            * get_group_node(param): Return group mean distribution for param.
            * get_var_node(param): Return group variability distribution for param.
            * get_subj_node(param): Return subject distribution for param.
            * get_bottom_node(param, params): Return distribution
                  for nodes at the bottom of the hierarchy param (e.g. the model
                  likelihood). params contains the associated model
                  parameters.

        In addition, the variable self.params must be defined as a
        list of Paramater().

    """

    def __init__(self, data, is_group_model=None, depends_on=None, trace_subjs=True,
                 plot_subjs=False, plot_var=False, include=(), replace_params=None):
        # Init
        self.include = set(include)

        self.nodes = {}
        self.mc = None
        self.trace_subjs = trace_subjs
        self.plot_subjs = plot_subjs
        self.plot_var = plot_var

        # Add data_idx field to data. Since we are restructuring the
        # data, this provides a means of getting the data out of
        # kabuki and sorting to according to data_idx to get the
        # original order.
        assert('data_idx' not in data.dtype.names),'A field named data_idx was found in the data file, please change it.'
        new_dtype = data.dtype.descr + [('data_idx', '<i8')]
        new_data = np.empty(data.shape, dtype=new_dtype)
        for field in data.dtype.fields:
            new_data[field] = data[field]
        new_data['data_idx'] = np.arange(len(data))
        data = new_data
        self.data = data

        if not depends_on:
            self.depends_on = {}
        else:
            # Support for supplying columns as a single string
            # -> transform to list
            for key in depends_on:
                if type(depends_on[key]) is str:
                    depends_on[key] = [depends_on[key]]
            # Check if column names exist in data
            for depend_on in depends_on.itervalues():
                for elem in depend_on:
                    if elem not in self.data.dtype.names:
                        raise KeyError, "Column named %s not found in data." % elem
            self.depends_on = depends_on
        self.depends_dict = OrderedDict()

        if is_group_model is None:
            if 'subj_idx' in data.dtype.names:
                if len(np.unique(data['subj_idx'])) != 1:
                    self.is_group_model = True
                else:
                    self.is_group_model = False
            else:
                self.is_group_model = False

        else:
            if is_group_model:
                if 'subj_idx' not in data.dtype.names:
                    raise ValueError("Group models require 'subj_idx' column in input data.")

            self.is_group_model = is_group_model

        # Should the model incorporate multiple subjects
        if self.is_group_model:
            self._subjs = np.unique(data['subj_idx'])
            self._num_subjs = self._subjs.shape[0]

        #set Parameters
        self.params = self.get_params()

        if replace_params != None:
            self.set_user_params(replace_params)

        self.params_dict = {}
        for param in self.params:
            self.params_dict[param.name] = param


    def set_user_params(self, replace_params):
        """replace parameters with user defined parameters"""
        if isinstance(replace_params, Parameter):
            replace_params = [replace_params]
        for new_param in replace_params:
            for i in range(len(self.params)):
                if self.params[i].name == new_param.name:
                    self.params[i] = new_param


    def _get_data_depend(self):
        """Partition data according to self.depends_on.

        :Returns:
            List of tuples with the data, the corresponding parameter
            distribution and the parameter name.

        """

        params = {} # use subj parameters to feed into model
        # Create new params dict and copy over nodes
        for name, param in self.params_include.iteritems():
            # Bottom nodes are created later
            if name in self.depends_on or param.is_bottom_node:
                continue
            if self.is_group_model and param.create_subj_nodes:
                params[name] = param.subj_nodes['']
            else:
                params[name] = param.group_nodes['']

        depends_on = copy(self.depends_on)

        # Make call to recursive function that does the partitioning
        data_dep = self._get_data_depend_rec(self.data, depends_on, params, [])

        return data_dep

    def _get_data_depend_rec(self, data, depends_on, params, dep_name, param=None):
        """Recursive function to partition data and params according
        to depends_on.

        """

        # Unfortunately, this function is quite complex as it
        # recursively parcels the data.

        if len(depends_on) != 0: # If depends are present
            data_params = []
            # Get first param from depends_on
            param_name = depends_on.keys()[0]
            col_name = depends_on.pop(param_name) # Take out param
            depend_elements = np.unique(data[col_name])
            # Loop through unique elements
            for depend_element in depend_elements:
                # Append dependent element name.
                dep_name.append(depend_element)
                # Extract rows containing unique element
                data_dep = data[data[col_name] == depend_element]

                # Add a key that is only the col_name that links to
                # the correct dependent nodes. This is the central
                # trick so that later on the get_bottom_node can use
                # params[col_name] and the bottm node will get linked to
                # the correct nodes automatically.
                param = self.params_include[param_name]

                # Add the node
                if self.is_group_model and param.create_subj_nodes:
                    params[param_name] = param.subj_nodes[str(depend_element)]
                else:
                    params[param_name] = param.group_nodes[str(depend_element)]
                # Recursive call with one less dependency and the selected data.
                data_param = self._get_data_depend_rec(data_dep,
                                                       depends_on=copy(depends_on),
                                                       params=copy(params),
                                                       dep_name=copy(dep_name),
                                                       param=param)
                data_params += data_param
                # Remove last item (otherwise we would always keep
                # adding the dep elems of in one column)
                dep_name.pop()
            return data_params

        else: # Data does not depend on anything (anymore)

            #create_
            if len(dep_name) != 0:
                if len(dep_name) == 1:
                    dep_name_str = str(dep_name[0])
                else:
                    dep_name_str = str(dep_name)
            else:
                dep_name_str = ''

            return [(data, params, dep_name, dep_name_str)]

    def create_nodes(self, max_retries=8):
        """Set group level distributions. One distribution for each
        parameter.

        :Arguments:
            retry : int
                How often to retry when model creation
                failed (due to bad starting values).

        """
        def _create():
            for name, param in self.params_include.iteritems():
                # Bottom nodes are created elsewhere
                if param.is_bottom_node:
                    continue
                # Check if parameter depends on data
                if name in self.depends_on.keys():
                    self._set_dependent_param(param)
                else:
                    self._set_independet_param(param)

            # Init bottom nodes
            for param in self.params_include.itervalues():
                if not param.is_bottom_node:
                    continue
                self._set_bottom_nodes(param, init=True)

            # Create bottom nodes
            for param in self.params_include.itervalues():
                if not param.is_bottom_node:
                    continue
                self._set_bottom_nodes(param, init=False)

        # Include all defined parameters by default.
        self.non_optional_params = [param.name for param in self.params if not param.optional]

        # Create params dictionary
        self.params_dict = OrderedDict()
        for param in self.params:
            self.params_dict[param.name] = param

        self.params_include = OrderedDict()
        for param in self.params:
            if param.name in self.include or not param.optional:
                self.params_include[param.name] = param

        for tries in range(max_retries):
            try:
                _create()
            except (pm.ZeroProbability, ValueError) as e:
                continue
            break
        else:
            print "After %f retries, still not good fit found." %(tries)
            raise e


        # Create model dictionary
        self.nodes = {}
        self.group_nodes = {}
        self.var_nodes = {}
        self.subj_nodes = {}
        self.bottom_nodes = {}

        for name, param in self.params_include.iteritems():
            for tag, node in param.group_nodes.iteritems():
                self.nodes[name+tag+'_group'] = node
                self.group_nodes[name+tag] = node
            for tag, node in param.subj_nodes.iteritems():
                self.nodes[name+tag+'_subj'] = node
                self.subj_nodes[name+tag] = node
            for tag, node in param.var_nodes.iteritems():
                self.nodes[name+tag+'_var'] = node
                self.var_nodes[name+tag] = node
            for tag, node in param.bottom_nodes.iteritems():
                self.nodes[name+tag+'_bottom'] = node
                self.bottom_nodes[name+tag] = node

        return self.nodes

    def map(self, runs=2, warn_crit=5, **kwargs):
        """
        Find MAP and set optimized values to nodes.

        :Arguments:
            runs : int
                How many runs to make with different starting values
            warn_crit: float
                How far must the two best fitting values be apart in order to print a warning message

        :Returns:
            pymc.MAP object of model.

        :Note:
            Forwards additional keyword arguments to pymc.MAP().

        """

        from operator import attrgetter

        maps = []

        for i in range(runs):
            # (re)create nodes to get new initival values
            self.create_nodes()
            m = pm.MAP(self.nodes)
            m.fit(**kwargs)
            print m.logp
            maps.append(m)

        # We want to use values of the best fitting model
        sorted_maps = sorted(maps, key=attrgetter('logp'))
        max_map = sorted_maps[-1]

        # If maximum logp values are not in the same range, there
        # could be a problem with the model.
        if runs >= 2:
            abs_err = np.abs(sorted_maps[-1].logp - sorted_maps[-2].logp)
            if abs_err > warn_crit:
                print "Warning! Two best fitting MAP estimates are %f apart. Consider using more runs to avoid local minima." % abs_err

        # Set values of nodes
        for name, node in max_map._dict_container.iteritems():
            if type(node) is pm.ArrayContainer:
                for i,subj_node in enumerate(node):
                    if not subj_node.observed:
                        self.nodes[name][i].value = subj_node.value
            elif not node.observed:
                self.nodes[name].value = node.value

        return max_map

    def mcmc(self, *args, **kwargs):
        """
        Returns pymc.MCMC object of model.

        :Note:
            Forwards arguments to pymc.MCMC().

        """

        if not self.nodes:
            self.create_nodes()

        self.mc = pm.MCMC(self.nodes, *args, **kwargs)

        return self.mc

    def sample(self, *args, **kwargs):
        """Sample from posterior.

        :Note:
            Forwards arguments to pymc.MCMC.sample().

        """

        # init mc if needed
        if self.mc == None:
            self.mcmc()

        # suppress annoying warnings
        if ('hdf5' in dir(pm.database)) and \
           isinstance(self.mc.db, pm.database.hdf5.Database):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', pm.database.hdf5.tables.NaturalNameWarning)

        # sample
        self.mc.sample(*args, **kwargs)

        return self.mc


    def print_group_stats(self, fname=None):
        stats_str = kabuki.analyze.gen_group_stats(self.stats())
        if fname is None:
            print stats_str
        else:
            with open(fname) as fd:
                fd.write(stats_str)


    def print_stats(self, fname=None):
        stats_str = kabuki.analyze.gen_stats(self.stats())
        if fname is None:
            print stats_str
        else:
            with open(fname) as fd:
                fd.write(stats_str)


    def _set_dependent_param(self, param):
        """Set parameter that depends on data.

        :Arguments:
            param_name : string
                Name of parameter that depends on data for
                which to set distributions.

        """

        # Get column names for provided param_name
        depends_on = self.depends_on[param.name]

        # Get unique elements from the columns
        data_dep = self.data[depends_on]
        uniq_data_dep = np.unique(data_dep)

        #update depends_dict
        self.depends_dict[param.name] = uniq_data_dep

        # Loop through unique elements
        for uniq_date in uniq_data_dep:
            # Select data
            data_dep_select = self.data[(data_dep == uniq_date)]

            # Create name for parameter
            tag = str(uniq_date)

            # Create parameter distribution from factory
            param.tag = tag
            param.data = data_dep_select
            if param.create_group_node:
                param.group_nodes[tag] = self.get_group_node(param)
            else:
                param.group_nodes[tag] = None
            param.reset()

            if self.is_group_model and param.create_subj_nodes:
                # Create appropriate subj parameter
                self._set_subj_nodes(param, tag, data_dep_select)

        return self

    def _set_independet_param(self, param):
        """Set parameter that does _not_ depend on data.

        :Arguments:
            param_name : string
                Name of parameter.

        """

        # Parameter does not depend on data
        # Set group parameter
        param.tag = ''
        if param.create_group_node:
            param.group_nodes[''] = self.get_group_node(param)
        else:
            param.group_nodes[''] = None
        param.reset()

        if self.is_group_model and param.create_subj_nodes:
            self._set_subj_nodes(param, '', self.data)

        return self

    def _set_subj_nodes(self, param, tag, data):
        """Set nodes with a parent.

        :Arguments:
            param_name : string
                Name of parameter.
            tag : string
                Element name.
            data : numpy.recarray
                Part of the data the parameter
                depends on.

        """
        # Generate subj variability parameter var
        param.tag = 'var'+tag
        param.data = data
        if param.create_group_node:
            param.var_nodes[tag] = self.get_var_node(param)
        else:
            param.var_nodes[''] = None
        param.reset()

        # Init
        param.subj_nodes[tag] = np.empty(self._num_subjs, dtype=object)
        # Create subj parameter distribution for each subject
        for subj_idx, subj in enumerate(self._subjs):
            data_subj = data[data['subj_idx']==subj]
            param.data = data_subj
            param.group = param.group_nodes[tag]
            if param.create_group_node:
                param.var = param.var_nodes[tag]
            param.tag = tag
            param.idx = subj_idx
            param.subj_nodes[tag][subj_idx] = self.get_subj_node(param)
            param.reset()

        return self

    def _set_bottom_nodes(self, param, init=False):
        """Set parameter node that has no parent.

        :Arguments:
            param_name : string
                Name of parameter.

        :Optional:
            init : bool
                Initialize parameter.

        """
        # Divide data and parameter distributions according to self.depends_on
        data_dep = self._get_data_depend()

        # Loop through parceled data and params and create an observed stochastic
        for i, (data, params_dep, dep_name_list, dep_name_str) in enumerate(data_dep):
            dep_name = dep_name_str
            if init:
                if self.is_group_model and param.create_subj_nodes:
                    param.bottom_nodes[dep_name] = np.empty(self._num_subjs, dtype=object)
                else:
                    param.bottom_nodes[dep_name] = None
            else:
                self._create_bottom_node(param, data, params_dep, dep_name, i)

        return self

    def _create_bottom_node(self, param, data, params, dep_name, idx):
        """Create parameter node object which has no parent.

        :Note:
            Called by self._set_bottom_node().

        :Arguments:
            param_name : string
                Name of parameter.
            data : numpy.recarray
                Data on which parameter depends on.
            params : list
                List of parameters the node depends on.
            dep_name : str
                Element name the node depends on.
            idx : int
                Subject index.

        """

        if self.is_group_model:
            for i, subj in enumerate(self._subjs):
                # Select data belonging to subj
                data_subj = data[data['subj_idx'] == subj]
                # Skip if subject was not tested on this condition
                if len(data_subj) == 0:
                    continue

                ########################################
                # Unfortunately, this is a little hairy since we have
                # to find the nodes of the right subject and the right
                # condition.

                # Here we'll store all nodes belonging to the subject
                selected_subj_nodes = {}
                # Find and store corresponding nodes
                for selected_param in self.params_include.itervalues():
                    # Since groupless nodes are not created in this function we
                    # have to search for the correct node and include it in
                    # the params.
                    if selected_param.is_bottom_node:
                        continue
                    if not selected_param.create_subj_nodes:
                        if selected_param.subj_nodes.has_key(dep_name):
                            selected_subj_nodes[selected_param.name] = selected_param.group_nodes[dep_name]
                        else:
                            selected_subj_nodes[selected_param.name] = params[selected_param.name]
                    else:
                        if selected_param.subj_nodes.has_key(dep_name):
                            selected_subj_nodes[selected_param.name] = selected_param.subj_nodes[dep_name][i]
                        else:
                            selected_subj_nodes[selected_param.name] = params[selected_param.name][i]

                # Set up param
                param.tag = dep_name
                param.idx = i
                param.data = data_subj
                # Call to the user-defined function!
                bottom_node = self.get_bottom_node(param, selected_subj_nodes)
                if bottom_node is not None and len(bottom_node.value) == 0:
                    print "Warning! Bottom node %s is not linked to data. Replacing with None." % param.full_name
                    param.bottom_nodes[dep_name][i] = None
                else:
                    param.bottom_nodes[dep_name][i] = bottom_node
                param.reset()
        else: # Do not use subj params, but group ones
            # Since group nodes are not created in this function we
            # have to search for the correct node and include it in
            # the params
            for selected_param in self.params_include.itervalues():
                if selected_param.subj_nodes.has_key(dep_name):
                    params[selected_param.name] = selected_param.subj_nodes[dep_name]

            if len(data) == 0:
                # If no data is present, do not create node.
                param.bottom_nodes[dep_name][i] = None
            else:
                param.tag = dep_name
                param.data = data
                # Call to user-defined function
                bottom_node = self.get_bottom_node(param, params)
                if bottom_node is not None and len(bottom_node.value) == 0:
                    print "Warning! Bottom node %s is not linked to data. Replacing with None." % param.full_name
                    param.bottom_nodes[dep_name] = None
                else:
                    param.bottom_nodes[dep_name] = bottom_node

            param.reset()

        return self


    def get_node(self, node_name, params):
        """Returns the node object with node_name from params if node
        is included in model, otherwise returns default value.

        """
        if node_name in self.include:
            return params[node_name]
        else:
            assert self.params_dict[node_name].default is not None, "Default value of not-included parameter not set."
            return self.params_dict[node_name].default


    def stats(self, *args, **kwargs):
        """
        smart call of MCMC.stats() for the model
        """
        try:
            nchains = self.mc.db.chains
        except AttributeError:
            raise ValueError("No model found.")

        #check which chain is going to be "stat"
        if 'chain' in kwargs:
            i_chain = kwargs['chain']
        else:
            i_chain = nchains

        #compute stats
        try:
            if self._stats_chain==i_chain:
                return self._stats
        except AttributeError:
            pass
        self._stats = self.mc.stats(*args, **kwargs)
        self._stats_chain = i_chain
        return self._stats


    def _set_traces(self, params, mc_model=None, add=False, chain=0):
        """Externally set the traces of group_params. This is needed
        when loading a model from a previous sampling run saved to a
        database.
        """
        if not mc_model:
            mc_model = self.mc

        # Loop through parameters and set traces
        for param_name, param_inst in params.iteritems():
            try:
                if add:
                    # Append trace
                    param_inst.trace._trace[chain] = np.concatenate((param_inst.trace._trace[chain],
                                                                     mc_model.trace(param_name)()))
                else:
                    param_inst.trace = mc_model.trace(param_name)
            except AttributeError: # param_inst is an array
                if self.trace_subjs:
                    for i, subj_param in enumerate(param_inst):
                        if add:
                            subj_param.trace._trace[chain] = np.concatenate((subj_param.trace._trace[chain],
                                                                             mc_model.trace(subj_param.__name__))())
                        else:
                            subj_param.trace = mc_model.trace(subj_param.__name__)

    def load_db(self, dbname, verbose=0, db_loader=None):
        """Load samples from a database created by an earlier model
        run (e.g. by calling .mcmc(dbname='test'))
        """
        if db_loader is None:
            db_loader = pm.database.sqlite.load

        # Set up model
        if not self.nodes:
            self.create_nodes()

        # Open database
        db = db_loader(dbname)

        # Create mcmc instance reading from the opened database
        self.mc = pm.MCMC(self.nodes, db=db, verbose=verbose)

        # Not sure if this does anything useful, but calling for good luck
        self.mc.restore_sampler_state()

        # Take the traces from the database and feed them into our
        # distribution variables (needed for _gen_stats())
        self._set_traces(self.group_nodes)

        if self.is_group_model:
            self._set_traces(self.var_nodes)
            self._set_traces(self.subj_nodes)

        return self

    #################################
    # Methods that can be overwritten
    #################################
    def get_group_node(self, param):
        """Create and return a uniform prior distribution for group
        parameter 'param'.

        This is used for the group distributions.

        """
        return pm.Uniform(param.full_name,
                          lower=param.lower,
                          upper=param.upper,
                          value=param.init,
                          verbose=param.verbose)

    def get_var_node(self, param):
        """Create and return a Uniform prior distribution for the
        variability parameter 'param'.

        Note, that we chose a Uniform distribution rather than the
        more common Gamma (see Gelman 2006: "Prior distributions for
        variance parameters in hierarchical models").

        This is used for the variability fo the group distribution.

        """
        return pm.Uniform(param.full_name, lower=param.var_lower, upper=param.var_upper,
                          value=.3, plot=self.plot_var)

    def get_subj_node(self, param):
        """Create and return a Truncated Normal distribution for
        'param' centered around param.group with standard deviation
        param.var and initialization value param.init.

        This is used for the individual subject distributions.

        """
        return pm.TruncatedNormal(param.full_name,
                                  a=param.lower,
                                  b=param.upper,
                                  mu=param.group,
                                  tau=param.var**-2,
                                  plot=self.plot_subjs,
                                  trace=self.trace_subjs,
                                  value=param.init)

    def init_from_existing_model(self, pre_model, step_method, **kwargs):
        """
        initialize the value and step methods of the model using an existing model
        """
        if not self.nodes:
            self.mcmc(**kwargs)
        all_nodes = list(self.mc.stochastics)
        all_pre_nodes = list(pre_model.mc.stochastics)
        assigned_values = 0
        assigned_steps = 0

        #loop over all nodes
        for i_node in range(len(all_nodes)):
            #get name of current node
            t_name = all_nodes[i_node].__name__
            #get the matched node from the pre_model
            pre_node = [x for x in all_pre_nodes if x.__name__ == t_name]
            if len(pre_node)==0:
                continue
            pre_node = pre_node[0]
            assigned_values += 1

            all_nodes[i_node].value= pre_node.value
            if step_method:
                step_pre_node = pre_model.mc.step_method_dict[pre_node][0]
                pre_sd = step_pre_node.proposal_sd * step_pre_node.adaptive_scale_factor
                if type(step_pre_node) == pm.Metropolis:
                    self.mc.use_step_method(pm.Metropolis(all_nodes[i_node],
                                                          proposal_sd = pre_sd))
                    assigned_steps += 1

        print "assigned values to %d nodes (out of %d)." % (assigned_values, len(all_nodes))
        if step_method:
            print "assigned step methods to %d (out of %d)." % (assigned_steps, len(all_nodes))

    def plot_posteriors(self):
        pm.Matplot.plot(self.mc)

    def subj_by_subj_map_init(self, runs=2, **map_kwargs):
        """
        initializing nodes by finding the MAP for each subject separately
        Input:
            runs - number of MAP runs for each subject
            map_kwargs - other arguments that will be passes on to the map function

        Note: This function should be run prior to the nodes creation, i.e.
        before running mcmc() or map()
        """

        #check if nodes were created. if they were it cause problems for deepcopy
        assert (not self.nodes), "function should be used before nodes are initialized."

        #init
        subjs = self._subjs
        n_subjs = len(subjs)

        empty_s_model = deepcopy(self)
        empty_s_model.is_group_model = False
        del empty_s_model._num_subjs, empty_s_model._subjs, empty_s_model.data

        self.create_nodes()

        #loop over subjects
        for i_subj in range(n_subjs):
            #create and fit single subject
            print "*!*!* fitting subject %d *!*!*" % subjs[i_subj]
            t_data = self.data[self.data['subj_idx'] == subjs[i_subj]]
            t_data = rec_drop_fields(t_data, ['data_idx'])
            s_model = deepcopy(empty_s_model)
            s_model.data = t_data
            s_model.map(method='fmin_powell', runs=runs, **map_kwargs)

            # copy to original model
            for (name, node) in s_model.group_nodes.iteritems():
                self.subj_nodes[name][i_subj].value = node.value

        #set group and var nodes
        for (param_name, d) in self.params_dict.iteritems():
            for (tag, nodes) in d.subj_nodes.iteritems():
                subj_values = [x.value for x in nodes]
                #set group node
                if d.group_nodes:
                    d.group_nodes[tag].value = np.mean(subj_values)
                #set var node
                if d.var_nodes:
                    if d.var_type == 'std':
                        d.var_nodes[tag].value = np.std(subj_values)
                    elif d.var_type == 'precision':
                        d.var_nodes[tag].value = np.std(subj_values)**-2
                    elif d.var_type == 'sample_size':
                        v = np.var(subj_values)
                        m = np.mean(subj_values)
                        d.var_nodes[tag].value = (m * (1 - m)) / v - 1
                    else:
                        raise ValueError, "unknown var_type"

