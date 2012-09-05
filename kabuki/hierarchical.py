 #!/usr/bin/python
from __future__ import division
from copy import copy

import numpy as np
from scipy.optimize import fmin_powell

from collections import OrderedDict, defaultdict

import pandas as pd
import pymc as pm
import warnings

from kabuki.utils import flatten

class Knode(object):
    def __init__(self, pymc_node, name, depends=(), col_name='', subj=False, hidden=False, **kwargs):
        self.pymc_node = pymc_node
        self.name = name
        self.kwargs = kwargs
        self.subj = subj
        self.col_name = col_name
        self.nodes = OrderedDict()
        self.hidden = hidden

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

        self.observed = 'observed' in kwargs

    def __repr__(self):
        return self.name

    def set_data(self, data):
        self.data = data

    def get_parent_depends(self):
        """returns the depends of the parents"""
        union_parent_depends = set()
        for name, parent in self.parents.iteritems():
            union_parent_depends.update(set(parent.depends))
        return union_parent_depends

    def init_nodes_db(self):
        data_col_names = list(self.data.columns)
        node_descriptors = ['knode_name', 'stochastic', 'observed', 'subj', 'node', 'tag', 'depends', 'hidden']
        stats = ['mean', 'std', '2.5q', '25q', '50q', '75q', '97.5q', 'mc err']

        columns = node_descriptors + data_col_names + stats

        # create central dataframe
        self.nodes_db = pd.DataFrame(columns=columns)

    def append_node_to_db(self, node, uniq_elem):
        #create db entry for knode
        row = {}
        row['knode_name'] = self.name
        row['observed'] = self.observed
        row['stochastic'] = isinstance(node, pm.Stochastic) and not self.observed
        row['subj'] = self.subj
        row['node'] = node
        row['tag'] = self.create_tag_and_subj_idx(self.depends, uniq_elem)[0]
        row['depends'] = self.depends
        row['hidden'] = self.hidden

        row = pd.DataFrame(data=[row], columns=self.nodes_db.columns, index=[node.__name__])

        for dep, elem in zip(self.depends, uniq_elem):
            row[dep] = elem

        self.nodes_db = self.nodes_db.append(row)

    def create(self):
        """create the pymc nodes"""

        self.init_nodes_db()

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
            tag, subj_idx = self.create_tag_and_subj_idx(self.depends, uniq_elem)
            node_name = self.create_node_name(tag, subj_idx=subj_idx)

            #get value for observed node
            if self.observed:
                kwargs['value'] = grouped_data[self.col_name].values

            # Deterministic nodes require a parent argument that is a
            # dict mapping parent names to parent nodes. Knode wraps
            # this; so here we have to fish out the parent nodes from
            # kwargs, put them into a parent dict and put that back
            # into kwargs, which will make pm.Determinstic() get a
            # parent dict as an argument.
            if self.pymc_node is pm.Deterministic:
                parents_dict = {}
                for name, parent in self.parents.iteritems():
                    parents_dict[name] = parent.get_node(self.depends, uniq_elem)
                    kwargs.pop(name)
                kwargs['parents'] = parents_dict


            # Deterministic nodes require a doc kwarg, we don't really
            # need that so if its not supplied, just use the name
            if self.pymc_node is pm.Deterministic and 'doc' not in kwargs:
                kwargs['doc'] = node_name

            #actually create the node
            node = self.pymc_node(name=node_name, **kwargs)

            self.nodes[uniq_elem] = node

            self.append_node_to_db(node, uniq_elem)


    def create_tag_and_subj_idx(self, cols, uniq_elem):
        uniq_elem = pd.Series(uniq_elem, index=cols)

        if 'subj_idx' in cols:
            subj_idx = uniq_elem['subj_idx']
            tag = uniq_elem.drop(['subj_idx']).values
        else:
            tag = uniq_elem.values
            subj_idx = None

        return tuple(tag), subj_idx


    def create_node_name(self, tag, subj_idx=None):
        # construct string that will become the node name
        s = self.name
        if len(tag) > 0:
            elems_str = '.'.join([str(elem) for elem in tag])
            s += "({elems})".format(elems=elems_str)
        if subj_idx is not None:
            s += ".{subj_idx}".format(subj_idx=subj_idx)

        return s

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

def intersect(t1, t2):
    # Preserves order, unlike set.
    return tuple([i for i in t2 if i in t1])

def test_subset_tuple():
    assert intersect(('a', 'b' , 'c'), ('a',)) == ('a',)
    assert intersect(('a', 'b' , 'c'), ('a', 'b')) == ('a', 'b')
    assert intersect(('a', 'b' , 'c'), ('a', 'c')) == ('a', 'c')
    assert intersect(('a', 'b' , 'c'), ('b', 'c')) == ('b', 'c')
    assert intersect(('c', 'b', 'a'), ('b', 'c')) == ('b', 'c')


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

        In addition, the variable self.params must be defined as a
        list of Paramater().

    """

    def __init__(self, data, is_group_model=None, depends_on=None, trace_subjs=True,
                 plot_subjs=False, plot_var=False):

        # Init
        self.plot_subjs = plot_subjs

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

        # create knodes (does not build according pymc nodes)
        self.knodes = self.create_knodes()

        #add data to knodes
        for knode in self.knodes:
            knode.set_data(self.data)

        # constructs pymc nodes etc and connects them appropriately
        self.create_model()


    def create_knodes(self):
        raise NotImplementedError("create_knodes has to be overwritten")


    def create_model(self, max_retries=8):
        """Set group level distributions. One distribution for each
        parameter.

        :Arguments:
            retry : int
                How often to retry when model creation
                failed (due to bad starting values).
        """

        def _create():
            for knode in self.knodes:
                knode.create()

        for tries in range(max_retries):
            try:
                _create()
            except (pm.ZeroProbability, ValueError):
                continue
            break
        else:
            print "After %f retries, still no good fit found." %(tries)
            _create()

        # create node container
        self.create_nodes_db()

        # Check whether all user specified column names (via depends_on) where used by the depends_on.
        assert set(flatten(self.depends.values())).issubset(set(flatten(self.nodes_db.depends))), "One of the column names specified via depends_on was not picked up. Check whether you specified the correct parameter value."

    def create_nodes_db(self):
        self.nodes_db = pd.concat([knode.nodes_db for knode in self.knodes])


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
            if i != 0:
                self.create_model()

            m = pm.MAP(self.nodes_db.node.values)
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


    def mcmc(self, assign_step_methods=True, *args, **kwargs):
        """
        Returns pymc.MCMC object of model.

        Input:
            assign_step_metheds <bool> : assign the step methods in params to the nodes

            The rest of the arguments are forwards to pymc.MCMC
        """

        self.mc = pm.MCMC(self.nodes_db.node.values, *args, **kwargs)

        self.pre_sample()

        return self.mc

    def pre_sample(self):
        pass


    def _assign_spx(self, param, loc, scale):
        """assign spx step method to param"""
        # TODO Imri
        # self.mc.use_step_method(kabuki.steps.SPXcentered,
        #                         loc=loc,
        #                         scale=scale,
        #                         loc_step_method=param.group_knode.step_method,
        #                         loc_step_method_args=param.group_knode.step_method_args,
        #                         scale_step_method=param.var_knode.step_method,
        #                         scale_step_method_args=param.var_knode.step_method_args,
        #                         beta_step_method=param.subj_knode.step_method,
        #                         beta_step_method_args=param.subj_knode.step_method_args)
        pass


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
                fd.write("\nDIC: %f\n" % self.mc.dic)
                fd.write("deviance: %f\n" % info['deviance'])
                fd.write("pD: %f" % info['pD'])


    def print_stats(self, fname=None, print_hidden=False, **kwargs):
        """print statistics of all variables
        Input (optional)
            fname <string> - the output will be written to a file named fname
            print_hidden <bool>  - print statistics of hidden nodes
        """
        self.append_stats_to_nodes_db()

        sliced_db = self.nodes_db.copy()

        # only print stats of stochastic, non-observed nodes
        if not print_hidden:
            sliced_db = sliced_db[(sliced_db['observed'] == False) & (sliced_db['hidden'] == False)]
        else:
            sliced_db = sliced_db[(sliced_db['observed'] == False)]

        stat_cols  = ['mean', 'std', '2.5q', '25q', '50q', '75q', '97.5q', 'mc err']

        for node_property, value in kwargs.iteritems():
            sliced_db = sliced_db[sliced_db[node_property] == value]

        sliced_db = sliced_db[stat_cols]

        self._output_stats(sliced_db.to_string(), fname)


    def get_node(self, node_name, params):
        """Returns the node object with node_name from params if node
        is included in model, otherwise returns default value.

        """
        if node_name in self.include:
            return params[node_name]
        else:
            assert self.param_container.params_dict[node_name].default is not None, "Default value of not-included parameter not set."
            return self.param_container.params_dict[node_name].default


    def append_stats_to_nodes_db(self, *args, **kwargs):
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

        #see if stats have been cached for this chain
        try:
            if self._stats_chain == i_chain:
                return
        except AttributeError:
            pass

        #update self._stats
        self._stats = self.mc.stats(*args, **kwargs)
        self._stats_chain = i_chain

        #add/overwrite stats to nodes_db
        for name, i_stats in self._stats.iteritems():
            self.nodes_db['mean'][name]   = i_stats['mean']
            self.nodes_db['std'][name]    = i_stats['standard deviation']
            self.nodes_db['2.5q'][name]   = i_stats['quantiles'][2.5]
            self.nodes_db['25q'][name]    = i_stats['quantiles'][25]
            self.nodes_db['50q'][name]    = i_stats['quantiles'][50]
            self.nodes_db['75q'][name]    = i_stats['quantiles'][75]
            self.nodes_db['97.5q'][name]  = i_stats['quantiles'][97.5]
            self.nodes_db['mc err'][name] = i_stats['mc error']


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
        # TODO Imri
        raise NotImplementedError("TODO")
        if not self.mc:
            self.mcmc(assign_step_methods=False, **mcmc_kwargs)

        pre_d = pre_model.param_container.stoch_by_tuple
        assigned_s = 0; assigned_v = 0

        #set the new nodes
        for (key, node) in self.param_container.stoch_by_tuple.iteritems():
            name, h_type, tag, idx = key
            if name not in pre_model.param_container.params:
                continue

            #if the key was found then match_nodes assigns the old node value to the new node
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


    def get_observeds(self):
        return self.nodes_db[self.nodes_db.observed == True]

    def iter_observeds(self):
        nodes = self.get_observeds()
        for node in nodes.iterrows():
            yield node

    def get_non_observeds(self):
        return self.nodes_db[self.nodes_db.observed == False]

    def iter_non_observeds(self):
        nodes = self.get_non_observeds()
        for node in nodes.iterrows():
            yield node

    def iter_stochastics(self):
        nodes = self.get_stochastics()
        for node in nodes.iterrows():
            yield node

    def get_stochastics(self):
        return self.nodes_db[self.nodes_db.stochastic == True]

    def get_subj_nodes(self, stochastic=True):
        select = (self.nodes_db['subj'] == True) & \
                 (self.nodes_db['stochastic'] == stochastic)

        return self.nodes_db[select]

    def iter_subj_nodes(self, **kwargs):
        nodes = self.get_subj_nodes(**kwargs)
        for node in nodes.iterrows():
            yield node

    def get_group_nodes(self, stochastic=True):
        select = (self.nodes_db['subj'] == False) & \
                 (self.nodes_db['stochastic'] == stochastic)

        return self.nodes_db[select]

    def iter_group_nodes(self, **kwargs):
        nodes = self.get_group_nodes(**kwargs)
        for node in nodes.iterrows():
            yield node

    @property
    def values(self):
        return {name: node['node'].value[()] for (name, node) in self.iter_non_observeds()}

    def _partial_optimize(self, optimize_nodes, evaluate_nodes):
        """Optimize part of the model.

        :Arguments:
            nodes : iterable
                list nodes to optimize.
        """
        non_observeds = [node for node in optimize_nodes if not node.observed]

        init_vals = [node.value for node in non_observeds]

        # define function to be optimized
        def opt(values):
            for value, node in zip(values, optimize_nodes):
                node.value = value
            try:
                logp_optimize = [node.logp for node in optimize_nodes]
                logp_evaluate = [node.logp for node in evaluate_nodes]
                return -np.sum(logp_optimize) - np.sum(logp_evaluate)
            except pm.ZeroProbability:
                return np.inf

        fmin_powell(opt, init_vals)

    def approximate_map(self):
        """Set model to its approximate MAP.
        """
        ###############################
        # In order to find the MAP of a hierarchical model one needs
        # to integrate over the subj nodes. Since this is difficult we
        # optimize the generations iteratively on the generation below.

        # only need this to get at the generations
        # TODO: Find out how to get this from pymc.utils.find_generations()
        m = pm.MCMC(self.nodes_db.node)
        generations = m.generations
        generations.append(self.get_observeds().node)

        for i in range(len(generations)-1, 0, -1):
            # Optimize the generation at i-1 evaluated over the generation at i
            self._partial_optimize(generations[i-1], generations[i])

        #update map in nodes_db
        self.nodes_db['map'] = np.NaN
        for name, value in self.values.iteritems():
            self.nodes_db['map'].ix[name] = value


    def create_family_normal(self, name, value=0, g_mu=None,
                             g_tau=15**-2, var_lower=1e-10,
                             var_upper=100, var_value=.1):
        if g_mu is None:
            g_mu = value

        knodes = OrderedDict()

        if self.is_group_model and name not in self.group_only_nodes:
            g = Knode(pm.Normal, '%s' % name, mu=g_mu, tau=g_tau,
                      value=value, depends=self.depends[name])
            var = Knode(pm.Uniform, '%s_var' % name, lower=var_lower,
                        upper=var_upper, value=var_value)
            tau = Knode(pm.Deterministic, '%s_tau' % name,
                        doc='%s_tau' % name, eval=lambda x: x**-2, x=var,
                        plot=False, trace=False, hidden=True)
            subj = Knode(pm.Normal, '%s_subj' % name, mu=g, tau=tau,
                         value=value, depends=('subj_idx',),
                         subj=True, plot=self.plot_subjs)
            knodes['%s'%name] = g
            knodes['%s_var'%name] = var
            knodes['%s_tau'%name] = tau
            knodes['%s_bottom'%name] = subj

        else:
            subj = Knode(pm.Normal, name, mu=g_mu, tau=g_tau,
                         value=value, depends=self.depends[name])

            knodes['%s_bottom'%name] = subj

        return knodes


    def create_family_trunc_normal(self, name, value=0, lower=None,
                                   upper=None, var_lower=1e-10,
                                   var_upper=100, var_value=.1):
        knodes = OrderedDict()

        if self.is_group_model and name not in self.group_only_nodes:
            g = Knode(pm.Uniform, '%s' % name, lower=lower,
                      upper=upper, value=value, depends=self.depends[name])
            var = Knode(pm.Uniform, '%s_var' % name, lower=var_lower,
                        upper=var_upper, value=var_value)
            tau = Knode(pm.Deterministic, '%s_tau' % name,
                        doc='%s_tau' % name, eval=lambda x: x**-2, x=var,
                        plot=False, trace=False, hidden=True)
            subj = Knode(pm.TruncatedNormal, '%s_subj' % name, mu=g,
                         tau=tau, a=lower, b=upper, value=value,
                         depends=('subj_idx',), subj=True, plot=self.plot_subjs)

            knodes['%s'%name] = g
            knodes['%s_var'%name] = var
            knodes['%s_tau'%name] = tau
            knodes['%s_bottom'%name] = subj

        else:
            subj = Knode(pm.Uniform, name, lower=lower,
                         upper=upper, value=value,
                         depends=self.depends[name])
            knodes['%s_bottom'%name] = subj

        return knodes


    def create_family_invlogit(self, name, value, g_mu=None, g_tau=15**-2,
                               var_lower=1e-10, var_upper=100, var_value=.1):
        if g_mu is None:
            g_mu = value

        # logit transform values
        value_trans = np.log(value) - np.log(1-value)
        g_mu_trans = np.log(g_mu) - np.log(1-g_mu)

        knodes = OrderedDict()

        if self.is_group_model and name not in self.group_only_nodes:
            g_trans = Knode(pm.Normal,
                            '%s_trans'%name,
                            mu=g_mu_trans,
                            tau=g_tau,
                            value=value_trans,
                            depends=self.depends[name],
                            plot=False,
                            hidden=True
            )

            g = Knode(pm.InvLogit, name, ltheta=g_trans, plot=True,
                      trace=True)

            var = Knode(pm.Uniform, '%s_var'%name, lower=var_lower,
                        upper=var_upper, value=var_value)

            tau = Knode(pm.Deterministic, '%s_tau'%name, doc='%s_tau'
                        % name, eval=lambda x: x**-2, x=var,
                        plot=False, trace=False, hidden=True)

            subj_trans = Knode(pm.Normal, '%s_subj_trans'%name,
                               mu=g_trans, tau=tau, value=value_trans,
                               depends=('subj_idx',), subj=True,
                               plot=False, hidden=True)

            subj = Knode(pm.InvLogit, '%s_subj'%name,
                         ltheta=subj_trans, depends=('subj_idx',),
                         plot=self.plot_subjs, trace=True, subj=True)

            knodes['%s_trans'%name]      = g_trans
            knodes['%s'%name]            = g
            knodes['%s_var'%name]        = var
            knodes['%s_tau'%name]        = tau

            knodes['%s_subj_trans'%name] = subj_trans
            knodes['%s_bottom'%name]     = subj

        else:
            g_trans = Knode(pm.Normal, '%s_trans'%name, mu=g_mu,
                            tau=g_tau, value=value_trans,
                            depends=self.depends[name], plot=False, hidden=True)

            g = Knode(pm.InvLogit, '%s'%name, ltheta=g_trans, plot=True,
                      trace=True )

            knodes['%s_trans'%name] = g_trans
            knodes['%s_bottom'%name] = g

        return knodes

    def create_family_exp(self, name, value=0, g_mu=None,
                          g_tau=15**-2, var_lower=1e-10, var_upper=100, var_value=.1):
        if g_mu is None:
            g_mu = value

        value_trans = np.log(value)
        g_mu_trans = np.log(g_mu)

        knodes = OrderedDict()
        if self.is_group_model and name not in self.group_only_nodes:
            g_trans = Knode(pm.Normal, '%s_trans' % name, mu=g_mu_trans,
                            tau=g_tau, value=value_trans,
                            depends=self.depends[name], plot=False, hidden=True)

            g = Knode(pm.Deterministic, '%s'%name, eval=lambda x: np.exp(x),
                      x=g_trans, plot=True)

            var = Knode(pm.Uniform, '%s_var' % name,
                        lower=var_lower, upper=var_upper, value=var_value)

            tau = Knode(pm.Deterministic, '%s_tau' % name, eval=lambda x: x**-2,
                        x=var, plot=False, trace=False, hidden=True)

            subj_trans = Knode(pm.Normal, '%s_subj_trans'%name, mu=g_trans,
                         tau=tau, value=value_trans, depends=('subj_idx',),
                         subj=True, plot=False, hidden=True)

            subj = Knode(pm.Deterministic, '%s_subj'%name, eval=lambda x: np.exp(x),
                         x=subj_trans,
                         depends=('subj_idx',), plot=self.plot_subjs,
                         trace=True, subj=True)

            knodes['%s_trans'%name]      = g_trans
            knodes['%s'%name]            = g
            knodes['%s_var'%name]        = var
            knodes['%s_tau'%name]        = tau
            knodes['%s_subj_trans'%name] = subj_trans
            knodes['%s_bottom'%name]     = subj

        else:
            g_trans = Knode(pm.Normal, '%s_trans' % name, mu=g_mu_trans,
                            tau=g_tau, value=value_trans,
                            depends=self.depends[name], plot=False, hidden=True)

            g = Knode(pm.Deterministic, '%s'%name, doc='%s'%name, eval=lambda x: np.exp(x), x=g_trans, plot=True)
            knodes['%s_trans'%name] = g_trans
            knodes['%s_bottom'%name] = g

        return knodes

