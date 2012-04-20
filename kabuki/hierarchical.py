 #!/usr/bin/python
from __future__ import division
from copy import copy

import numpy as np
import numpy.lib.recfunctions as rec

from collections import OrderedDict, defaultdict

import pandas as pd
import pymc as pm
import warnings

import kabuki
from copy import copy, deepcopy


# (defaultdict) param_name -> (defaultdict) column_names -> elems
#self.create_model(depends={'v':('col1')})

# defaultdict(lambda: defaultdict(lambda: ()))

class Knode(object):
    def __init__(self, pymc_node, name, depends=(), col_name='', subj=False, **kwargs):
        self.pymc_node = pymc_node
        self.name = name
        self.kwargs = kwargs
        self.subj = subj
        self.col_name = col_name
        self.nodes = OrderedDict()

        #create self.parents
        self.parents = {}
        for (name, value) in self.kwargs.iteritems():
            if isinstance(value, Knode):
                self.parents[name] = value

        # Create depends set and update based on parents' depends
        depends = set(depends)
        if self.subj:
            depends.add('subj_idx')
        depends.update(self.get_parent_depends())

        self.depends = sorted(list(depends))

        self.observed =  'observed' in kwargs

    def set_data(self, data):
        self.data = data

    def get_parent_depends(self):
        union_parent_depends = set()
        for name, parent in self.parents.iteritems():
            union_parent_depends.update(set(parent.depends))
        return union_parent_depends

    def create(self):

        #group data
        if len(self.depends) == 0:
            grouped = [((), self.data)]
        else:
            grouped = self.data.groupby(self.depends)

        #create all the knodes
        for uniq_elem, grouped_data in grouped:
            if not isinstance(uniq_elem, tuple):
                uniq_elem = (uniq_elem,)

            # create new kwargs to pass to the new pymc node
            kwargs = self.kwargs.copy()

            # update kwarg with the right parent
            for name, parent in self.parents.iteritems():
                kwargs[name] = parent.get_node(self.depends, uniq_elem)

            #get node name
            node_name = self.create_node_name(uniq_elem)

            #get value for observed node
            if self.observed:
                kwargs['value'] = grouped_data[self.col_name].values

            #treat deterministic nodes

            #actually create the node
            node = self.pymc_node(node_name, **kwargs)

            self.nodes[uniq_elem] = node

    def create_node_name(self, uniq_elem):
        # TODO
        return self.name + str(uniq_elem)


    def get_node(self, cols, elems):
        """Return the node that depends on the same elements.

        Called by the child to receive specific parent node.

        :Arguments:
            col_to_elem : dict
                Maps column names to elements.
                e.g. {'col1': 'elem1', 'col2': 'elem2', 'col3': 'elem3'}
        """

        col_to_elem = {}
        for col, elem in zip(cols, elems):
            col_to_elem[col] = elem

        # Find the column names that overlap with the ones we have
        overlapping_cols = intersect(cols, self.depends)

        # Create new tag for the specific elements we are looking for (that overlap)
        deps_on_elems = tuple([col_to_elem[col] for col in overlapping_cols])

        return self.nodes[deps_on_elems]


# in Hierarchical: self.create_model(): for... knode.set_data(self.data); knode.create()

def intersect(t1, t2):
    # Preserves order, unlike set.
    return tuple([i for i in t2 if i in t1])

def test_subset_tuple():
    assert get_overlapping_elements(('a', 'b' , 'c'), ('a',)) == ('a',)
    assert get_overlapping_elements(('a', 'b' , 'c'), ('a', 'b')) == ('a', 'b')
    assert get_overlapping_elements(('a', 'b' , 'c'), ('a', 'c')) == ('a', 'c')
    assert get_overlapping_elements(('a', 'b' , 'c'), ('b', 'c')) == ('b', 'c')
    assert get_overlapping_elements(('c', 'b', 'a'), ('b', 'c')) == ('b', 'c')

class ParameterContainer(object):
    def __init__(self, params, is_group_model, trace_subjs, plot_subjs, plot_var, include=()):
        """initialize self.params"""

        self.is_group_model = is_group_model
        self.include = include
        self.params = params

        # Initialize parameter dicts.
        self.group_nodes = OrderedDict()
        self.subj_nodes = OrderedDict()
        self.bottom_nodes = OrderedDict()

        self.nodes = None

        if replace_params == None:
            replace_params = ()

        #change the default parametrs
        self._change_default_params(replace_params, update_params)

        for param in self.params:
            #set has_x_node
            param.has_group_nodes = (param.group_knode is not None)
            param.has_var_nodes = (param.var_knode is not None)
            param.has_subj_nodes = (param.subj_knode is not None)

            #set other attributes
            if param.is_bottom_node:
                param.has_subj_nodes = True
                continue
            if not trace_subjs and param.has_subj_nodes:
                param.subj_knode.args['trace'] = False
            if not plot_subjs and param.has_subj_nodes:
                param.subj_knode.args['plot'] = False
            if not plot_var and param.has_var_nodes:
                param.var_knode.args['plot'] = False

        #set params_dict
        self.params_dict = {}
        for param in self.params:
            if param.name in self.include or not param.optional:
                param.include = True
            else:
                param.include = False
            self.params_dict[param.name] = param


    def iter_group_nodes(self):
        for name, node in self.group_nodes.iteritems():
            yield (name, node)

    def iter_var_nodes(self):
        for name, node in self.var_nodes.iteritems():
            yield (name, node)

    def iter_subj_nodes(self):
        for name, node in self.subj_nodes.iteritems():
            yield (name, node)

    def iter_bottom_nodes(self):
        for name, param in self.params_dict.iteritems():
            if not param.is_bottom_node:
                continue
            for tag, node in param.bottom_nodes.iteritems():
                yield name, tag, node

    def iter_bottom_params(self, include=True):
        for name, param in self.params_dict.iteritems():
            if not param.is_bottom_node or not param.include:
                continue

            yield name, param

    def iter_params(self, optional=False, include=True, bottom=False):
        for name, param in self.params_dict.iteritems():
            if optional and not param.optional:
                continue

            if include and not param.include:
                continue

            if not bottom and param.is_bottom_node:
                continue

            yield (name, param)


    def _change_default_params(self, replace_params, update_params):
        """replace/update parameters with user defined parameters"""
        #replace params
        if type(replace_params)==Parameter:
            replace_params = [replace_params]
        for new_param in replace_params:
            for i in range(len(self.params)):
                if self.params[i].name == new_param.name:
                    self.params[i] = new_param

        #update params
        if update_params == None:
            return

        for (param_name, dict) in update_params.iteritems():
            found_param = False
            for i in range(len(self.params)):
                if self.params[i].name == param_name:
                    for (key, new_value) in dict.iteritems():
                        if hasattr(self.params[i], key):
                            setattr(self.params[i], key, new_value)
                        else:
                            raise ValueError, "An invalid key (%s) was found in update_params for Parameter %s" % (key, param_name)
                    found_param = True
                    break
            if not found_param:
                raise ValueError, "An invalid parameter (%s) was found in update_params" % (param_name)


    def reinit(self):
        # Initialize parameter dicts.
        self.group_nodes = OrderedDict()
        self.var_nodes = OrderedDict()
        self.subj_nodes = OrderedDict()
        self.bottom_nodes = OrderedDict()
        self.nodes = OrderedDict()

        #dictionary of stochastics. the keys are names of single nodes
        self.stoch_by_name = {}
        #dictionary of stochastics. the keys are (param_name, h_type, tag, idx)
        #htype (string)= 'g':group, 'v':var, 's':subject
        #idx (int) - for subject nodes it's the subject idx, for other nodes its -1
        self.stoch_by_tuple = {}

        # go over all nodes in params and assign them to the dictionaries
        for name, param in self.iter_params(include=True):
            #group nodes
            for tag, node in param.group_nodes.iteritems():
                tag = str(tag)
                self.nodes[name+tag+'_group'] = node
                self.group_nodes[name+tag] = node
                self.stoch_by_tuple[(name,'g',tag,-1)] = node
                self.stoch_by_name[node.__name__] = node

            #var nodes
            for tag, node in param.var_nodes.iteritems():
                tag = str(tag)
                self.nodes[name+tag+'_var'] = node
                self.var_nodes[name+tag] = node
                self.stoch_by_tuple[(name,'v',tag,-1)] = node
                self.stoch_by_name[node.__name__] = node

            #subj nodes
            for tag, nodes in param.subj_nodes.iteritems():
                tag = str(tag)
                self.nodes[name+tag+'_subj'] = nodes
                self.subj_nodes[tag] = nodes
                for (idx, node) in enumerate(nodes):
                    self.stoch_by_tuple[(name,'s',tag,idx)] = node
                    self.stoch_by_name[node.__name__] = node

        #bottom nodes
        for name, tag, nodes in self.iter_bottom_nodes():
            tag = str(tag)
            self.nodes[name+tag+'_bottom'] = nodes
            self.bottom_nodes[name+tag] = nodes
            if self.is_group_model:
                for (idx, node) in enumerate(nodes):
                    self.stoch_by_tuple[(name,'b',tag,idx)] = node
                    self.stoch_by_name[name] = node
            else:
                self.stoch_by_tuple[(name,'b',tag,-1)] = nodes
                self.stoch_by_name[name] = nodes

        #update knodes
        for name, param in self.iter_params(include=True):
            if param.is_bottom_node:
                continue
            param.knodes['group'].nodes = param.group_nodes
            #try to update var knodes and subj knodes if they exist
            try:
                param.knodes['var'].nodes = param.var_nodes
                param.knodes['subj'].nodes = param.subj_nodes
            except AttributeError:
                pass

        return self.nodes


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

        update_params : dictionary that holds dictionaries
            User defined parameters that to update files in the default ones.
            the keys of the dictionary should be the names of the parameters that
            one wants to update. The values are another dictionary with keys for the
            attributes the will be updated to the associated values.
            e.g., to change parameter x's group_step_method to Metropolis
            one should pass the following
            {'x' : {'group_step_method': Metropolis}}


    :Note:
        This class must be inherited. The child class must provide
        the following functions:
            * get_bottom_node(param, params): Return distribution
                  for nodes at the bottom of the hierarchy param (e.g. the model
                  likelihood). params contains the associated model
                  parameters.

        In addition, the variable self.params must be defined as a
        list of Paramater().

    """

    def __init__(self, data, is_group_model=None, depends_on=None, trace_subjs=True,
                 plot_subjs=False, plot_var=False, include=()):

        # Init
        self.include = set(include)

        self.mc = None

        self.data = pd.DataFrame(data)

        if not depends_on:
            depends_on = {}
        else:
            # Support for supplying columns as a single string
            # -> transform to list
            for key in depends_on:
                if isinstance(depends_on[key], str):
                    depends_on[key] = [depends_on[key]]
            # Check if column names exist in data
            for depend_on in depends_on.itervalues():
                for elem in depend_on:
                    if elem not in self.data.columns:
                        raise KeyError, "Column named %s not found in data." % elem


        self.depends = defaultdict(lambda: ())
        for key, value in depends_on.iteritems():
            self.depends[key] = value


        # Determine if group model
        if is_group_model is None:
            if 'subj_idx' in self.data.columns:
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
        else:
            self._num_subjs = 1

        self.num_subjs = self._num_subjs

        # self.param_container = ParameterContainer(self.create_params(), is_group_model, trace_subjs, plot_subjs, plot_var, include=include)

        self.knodes = self.create_knodes()

        #add data to knodes
        for knode in self.knodes:
            knode.set_data(self.data)

        self.create_model()

    def create_model(self, max_retries=8):
        """Set group level distributions. One distribution for each
        parameter.

        :Arguments:
            retry : int
                How often to retry when model creation
                failed (due to bad starting values).
        """
        # TODO: max_retries is causing bugs, so it was commented out.
        #     we need to delete the nodes that are created using _create before the next retry.
        #     I would like to check if we actually need this option
        # TW: We definitely need it, model creation fails all the time. Wouldn't it be enough
        #     to just delete all the nodes then?

        def _create():
            # self.param_container.reinit()

            for knode in self.knodes:
                knode.create()

        for tries in range(max_retries):
            try:
                _create()
            except (pm.ZeroProbability, ValueError) as e:
                continue
            break
        else:
            print "After %f retries, still no good fit found." %(tries)
            _create()

    def map(self, runs=2, warn_crit=5, method='fmin_powell', **kwargs):
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

        # I.S: when using MAP with Hierarchical model the subjects nodes should be
        # integrated out before the computation of the MAP (see Pinheiro JC, Bates DM., 1995, 2000).
        # since we are not integrating we get a point estimation for each
        # subject which is not what we want.
        if self.is_group_model:
            raise NotImplementedError("""Sorry, This method is not yet implemented for group models.
            you might consider using the subj_by_subj_map_init method""")


        maps = []

        for i in range(runs):
            # (re)create nodes to get new initival values.
            #nodes are not created for the first iteration if they already exist
            if (i > 0) or (not self.param_container.nodes):
                self.create_nodes()

            m = pm.MAP(self.param_container.nodes)
            m.fit(method, **kwargs)
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
            if isinstance(node, pm.ArrayContainer):
                for i,subj_node in enumerate(node):
                    if isinstance(node, pm.Node) and not subj_node.observed:
                        self.param_container.nodes[name][i].value = subj_node.value
            elif isinstance(node, pm.Node) and not node.observed:
                self.param_container.nodes[name].value = node.value

        return max_map

    def _assign_spx(self, param, loc, scale):
        """assign spx step method to param"""
        self.mc.use_step_method(kabuki.steps.SPXcentered,
                                loc=loc,
                                scale=scale,
                                loc_step_method=param.group_knode.step_method,
                                loc_step_method_args=param.group_knode.step_method_args,
                                scale_step_method=param.var_knode.step_method,
                                scale_step_method_args=param.var_knode.step_method_args,
                                beta_step_method=param.subj_knode.step_method,
                                beta_step_method_args=param.subj_knode.step_method_args)


    def mcmc(self, assign_step_methods=True, *args, **kwargs):
        """
        Returns pymc.MCMC object of model.

        Input:
            assign_step_metheds <bool> : assign the step methods in params to the nodes

            The rest of the arguments are forwards to pymc.MCMC
        """

        if not self.param_container.nodes:
            self.create_nodes()

        nodes ={}
        for (name, value) in self.param_container.nodes.iteritems():
            if value != None:
                nodes[name] = value

        self.mc = pm.MCMC(nodes, *args, **kwargs)

        if assign_step_methods and self.is_group_model:
            self.mcmc_step_methods()

        return self.mc


    def mcmc_step_methods(self):
        #assign step methods
        for name, param in self.param_container.iter_params(include=True):
            #assign SPX when share_var
            if param.use_spx and param.share_var:
                loc = param.group_nodes.values()
                scale = param.var_nodes.values()[0]
                self._assign_spx(param, loc, scale)

            #assign SPX when var is not shared
            elif param.use_spx and not param.share_var:
                for (tag, node) in param.group_nodes.iteritems():
                    loc=param.group_nodes[tag]
                    scale=param.var_nodes[tag]
                    self._assign_spx(param, loc, scale)

            #assign other step methods
            else:
                for knode in param.knodes.itervalues():
                    #check if we need to assign step method
                    if (knode is None) or knode.step_method is None:
                        continue
                    #assign it to all the nodes in knode
                    for node in knode.nodes.itervalues():
                        step = knode.step_method
                        args = knode.step_method_args
                        if node is None:
                            continue
                        #check if it is a single node, otherwise it's a node array
                        elif isinstance(node, pm.Stochastic):
                            self.mc.use_step_method(step, node, **args)
                        else:
                            [self.mc.use_step_method(step, t_node, **args) for t_node in node]

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
            warnings.simplefilter('ignore', pm.database.hdf5.tables.NaturalNameWarning)

        # sample
        self.mc.sample(*args, **kwargs)

        return self.mc

    def dic_info(self):
        """returns information about the model DIC"""

        info = {}
        info['DIC'] = self.mc.dic
        info['deviance']  = np.mean(self.mc.db.trace('deviance')(), axis=0)
        info['pD'] = info['DIC'] - info['deviance']

        return info

    def _output_stats(self, stats_str, fname=None):
        """
        used by print_stats and print_group_stats to print the stats to the screen
        or to file
        """
        info = self.dic_info()
        if fname is None:
            print stats_str
            print "DIC: %f" % self.mc.dic
            print "deviance: %f" % info['deviance']
            print "pD: %f" % info['pD']
        else:
            with open(fname, 'w') as fd:
                fd.write(stats_str)
                fd.write("DIC: %f\n" % self.mc.dic)
                fd.write("deviance: %f\n" % info['deviance'])
                fd.write("pD: %f\n" % info['pD'])

    def print_group_stats(self, fname=None):
        """print statistics of group variables
        Input (optional)
            fname <string> - the output will be written to a file named fname
        """
        stats_str = kabuki.analyze.gen_group_stats(self.stats())
        self._output_stats(stats_str, fname)

    def print_stats(self, fname=None):
        """print statistics of all variables
        Input (optional)
            fname <string> - the output will be written to a file named fname
        """
        stats_str = kabuki.analyze.gen_stats(self.stats())
        self._output_stats(stats_str, fname)


    def get_node(self, node_name, params):
        """Returns the node object with node_name from params if node
        is included in model, otherwise returns default value.

        """
        if node_name in self.include:
            return params[node_name]
        else:
            assert self.param_container.params_dict[node_name].default is not None, "Default value of not-included parameter not set."
            return self.param_container.params_dict[node_name].default


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



    def load_db(self, dbname, verbose=0, db='sqlite'):
        """Load samples from a database created by an earlier model
        run (e.g. by calling .mcmc(dbname='test'))

        :Arguments:
            dbname : str
                File name of database
            verbose : int <default=0>
                Verbosity level
            db : str <default='sqlite'>
                Which database backend to use, can be
                sqlite, pickle, hdf5, txt.
        """


        if db == 'sqlite':
            db_loader = pm.database.sqlite.load
        elif db == 'pickle':
            db_loader = pm.database.pickle.load
        elif db == 'hdf5':
            db_loader = pm.database.hdf5.load
        elif db == 'txt':
            db_loader = pm.database.txt.load

        # Set up model
        if not self.param_container.nodes:
            self.create_nodes()

        # Ignore annoying sqlite warnings
        warnings.simplefilter('ignore', UserWarning)

        # Open database
        db = db_loader(dbname)

        # Create mcmc instance reading from the opened database
        self.mc = pm.MCMC(self.param_container.nodes, db=db, verbose=verbose)

        # Not sure if this does anything useful, but calling for good luck
        self.mc.restore_sampler_state()

        # Take the traces from the database and feed them into our
        # distribution variables (needed for _gen_stats())

        return self



    def init_from_existing_model(self, pre_model, assign_values=True, assign_step_methods=True,
                                 match=None, **mcmc_kwargs):
        """
        initialize the value and step methods of the model using an existing model
        Input:
            pre_model - existing mode

            assign_values (boolean) - should values of nodes from the existing model
                be assigned to the new model

            assign_step_method (boolean) - same as assign_values only for step methods

            match (dict) - dictionary which maps tags from the new model to tags from the
                existing model. match is a dictionary of dictionaries and it has
                the following structure:  match[name][new_tag] = pre_tag
                name is the parameter name. new_tag is the tag of the new model,
                and pre_tag is a single tag or list of tags from the exisiting model that will be map
                to the new_tag.
        """
        if not self.mc:
            self.mcmc(assign_step_methods=False, **mcmc_kwargs)

        pre_d = pre_model.param_container.stoch_by_tuple
        assigned_s = 0; assigned_v = 0

        #set the new nodes
        for (key, node) in self.param_container.stoch_by_tuple.iteritems():
            name, h_type, tag, idx = key
            if name not in pre_model.param_container.params:
                continue

            #if the key was found then match_nodes is the assign the old node value to the new node
            if pre_d.has_key(key):
                matched_nodes = [pre_d[key]]

            else: #match tags
                #get the matching pre_tags
                try:
                    pre_tags = match[name][tag]
                except TypeError, AttributeError:
                    raise ValueError('match argument does not have the coorect name or tag')

                if type(pre_tags) == str:
                    pre_tags = [pre_tags]

                #get matching nodes
                matched_nodes = [pre_d[(name, h_type, x, idx)] for x in pre_tags]

            #average matched_nodes values
            if assign_values:
                node.value = np.mean([x.value for x in matched_nodes])
                assigned_v += 1

            #assign step method
            if assign_step_methods:
                assigned_s += self._assign_step_methods_from_existing(node, pre_model, matched_nodes)

        print "assigned %d values (out of %d)." % (assigned_v, len(self.mc.stochastics))
        print "assigned %d step methods (out of %d)." % (assigned_s, len(self.mc.stochastics))


    def _assign_step_methods_from_existing(self, node, pre_model, matched_nodes):
        """
        private funciton used by init_from_existing_model to assign a node
        using matched_nodes from pre_model
        Output:
             assigned (boolean) - step method was assigned

        """

        if isinstance(matched_nodes, pm.Node):
            matched_node = [matched_nodes]

        #find the step methods
        steps = [pre_model.mc.step_method_dict[x][0] for x in matched_nodes]

        #only assign it if it's a Metropolis
        if isinstance(steps[0], pm.Metropolis):
            pre_sd = np.median([x.proposal_sd * x.adaptive_scale_factor for x in steps])
            self.mc.use_step_method(pm.Metropolis, node, proposal_sd = pre_sd)
            assigned = True
        else:
            assigned = False

        return assigned

    def plot_posteriors(self, parameters=None, plot_subjs=False, **kwargs):
        """
        plot the nodes posteriors
        Input:
            parameters (optional) - a list of parameters to plot.
            plot_subj (optional) - plot subjs nodes

        TODO: add attributes plot_subjs and plot_var to kabuki
        which will change the plot attribute in the relevant nodes
        """

        if parameters is None: #plot the model
            pm.Matplot.plot(self.mc, **kwargs)

        else: #plot only the given parameters

            if not isinstance(parameters, list):
                 parameters = [parameters]

            #get the nodes which will be plotted
            for param in parameters:
                nodes = tuple(np.unique(param.group_nodes.values() + param.var_nodes.values()))
                if plot_subjs:
                    for nodes_array in param.subj_nodes.values():
                        nodes += list(nodes_array)
            #this part does the ploting
            for node in nodes:
                plot_value = node.plot
                node.plot = True
                pm.Matplot.plot(node, **kwargs)
                node.plot = plot_value

    def subj_by_subj_map_init(self, runs=2, verbose=1, **map_kwargs):
        """
        initializing nodes by finding the MAP for each subject separately
        Input:
            runs - number of MAP runs for each subject
            map_kwargs - other arguments that will be passes on to the map function

        Note: This function should be run prior to the nodes creation, i.e.
        before running mcmc() or map()
        """

        # check if nodes were created. if they were it cause problems for deepcopy
        assert (not self.param_container.nodes), "function should be used before nodes are initialized."

        #init
        subjless = {}
        subjs = self._subjs
        n_subjs = len(subjs)
        empty_s_model = deepcopy(self)
        empty_s_model.is_group_model = False
        del empty_s_model.num_subjs, empty_s_model._subjs, empty_s_model.data

        self.create_nodes()

        # loop over subjects
        for i_subj in range(n_subjs):
            # create and fit single subject
            if verbose > 0: print "*!*!* fitting subject %d *!*!*" % subjs[i_subj]
            t_data = self.data[self.data['subj_idx'] == subjs[i_subj]]
            s_model = deepcopy(empty_s_model)
            s_model.data = t_data
            s_model.map(method='fmin_powell', runs=runs, **map_kwargs)

            # copy to original model
            for (name, node) in s_model.param_container.iter_group_nodes():
                #try to assign the value of the node to the original model
                try:
                    self.param_container.subj_nodes[name][i_subj].value = node.value
                #if it fails it mean the param has no subj nodes
                except KeyError:
                    if subjless.has_key(name):
                        subjless[name].append(node.value)
                    else:
                        subjless[name] = [node.value]

        #set group and var nodes for params with subjs
        for (param_name, param) in self.param_container.iter_params():
            #if param has subj nodes than compute group and var nodes from them
            if param.has_subj_nodes:
                for (tag, nodes) in param.subj_nodes.iteritems():
                    subj_values = [x.value for x in nodes]
                    #set group node
                    if param.has_group_nodes:
                        param.group_nodes[tag].value = np.mean(subj_values)
                    #set var node
                    if param.has_var_nodes:
                        param.var_nodes[tag].value = param.var_func(subj_values)

        #set group nodes of subjless nodes
        for (name, values) in subjless.iteritems():
            self.param_container.group_nodes[name].value = np.mean(subjless[name])
