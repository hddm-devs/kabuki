import kabuki
from kabuki.hierarchical import Parameter
import numpy as np
import unittest
import pymc as pm
import test_utils

def class_factory(num_params=1, create_group_node=True, create_subj_nodes=True):
    params_local = []
    for i_param in range(num_params):
        create_group_node = bool(round(np.random.rand()))
        if create_group_node:
            create_subj_nodes = bool(round(np.random.rand()))
        else:
            create_subj_nodes = True

        p = Parameter('test%i'%i_param, lower=1, upper=10, create_group_node=create_group_node, create_subj_nodes=True)
        params_local.append(p)

    params_local.append(Parameter('observed', is_bottom_node=True))

    class Test(kabuki.Hierarchical):
        def get_params(self):
            return params_local

        def get_bottom_node(self, param, params):
            return pm.Normal(param.full_name, mu=params['test0'], tau=1, value=param.data['score'], observed=True)

        def get_group_node(self, param):
            return pm.Uniform(param.full_name,
                              lower=param.lower,
                              upper=param.upper)


        def get_var_node(self, param):
            return pm.Uniform(param.full_name, lower=0., upper=10.,
                              value=.1, plot=self.plot_var)


        def get_subj_node(self, param):
            if param.create_group_node:
                return pm.Normal(param.full_name,
                                 mu=param.group,
                                 tau=param.var)
            else:
                return pm.Uniform(param.full_name, lower=0, upper=10)


    return Test

class TestHierarchical(unittest.TestCase):
    def runTest(self):
        pass

    def __init__(self, *args, **kwargs):
        super(TestHierarchical, self).__init__(*args, **kwargs)

        np.random.seed(31337)
        self.num_subjs = 5
        self.subjs = range(self.num_subjs)
        pts_per_subj = 100
        self.data = np.empty(self.num_subjs*pts_per_subj, dtype=([('subj_idx',np.int), ('score',np.float), ('dep', 'S8'), ('foo','S8')]))

        self.models = []
        for subj in self.subjs:
            self.data['subj_idx'][subj*pts_per_subj:(subj+1)*pts_per_subj] = subj
            self.data['score'][subj*pts_per_subj:(subj+1)*pts_per_subj] = np.random.randn(pts_per_subj)
            self.data['dep'][subj*pts_per_subj:(subj+1)*pts_per_subj] = 'dep1'
            self.data['dep'][subj*pts_per_subj+pts_per_subj/2:(subj+1)*pts_per_subj] = 'dep2'
            self.data['foo'][subj*pts_per_subj:(subj+1)*pts_per_subj] = 'bar1'
            self.data['foo'][subj*pts_per_subj+pts_per_subj/2:(subj+1)*pts_per_subj] = 'bar2'

    def setUp(self):
        for i in range(1,10):
            model_class = class_factory(num_params=i)
            self.models.append(model_class(self.data))
            self.models.append(model_class(self.data, depends_on={'test0':['dep']}))
            if i > 2:
                self.models.append(model_class(self.data, depends_on={'test0':['dep'], 'test1':['foo']}))
                self.models.append(model_class(self.data, depends_on={'test0':['dep'], 'test1':['dep','foo']}))

    def test_models(self):
        for model in self.models:
            self.tst_model(model)

    def tst_model(self, model):
        model.create_nodes()

        # Create param names that should have been created
        for param in model.params:
            print model.depends_on

            if param.name in model.depends_on.keys():
                deps = model.depends_on[param.name]
            elif param.is_bottom_node:
                vals = model.depends_on.values()
                if len(vals) > 1 and type(vals[0]) is list:
                    # Flatten
                    vals = sum(vals, [])
                deps = np.unique(np.array(vals))
            else:
                deps = []

            if len(deps)!=0:
                uniqs = np.unique(model.data[deps])
                if not param.is_bottom_node:
                    param_names = ["%s%s"%(param.name, dep) for dep in uniqs]
                else:
                    uniqs = [elem for elems in uniqs for elem in elems]
                    param_names = ["%s('%s',)"%(param.name, dep) for dep in uniqs]
            else:
                param_names = [param.name]

            # Each parameter can create multiple nodes if it depends on data
            for param_name in param_names:
                if param.create_group_node:
                    print "GROUP NODES"
                    # Check if group node exists
                    self.assertIn(param_name, model.group_nodes.keys())
                    self.assertIn(param_name+'_group', model.nodes.keys())

                    # Check node parents
                    self.assertIs(param.upper, model.group_nodes[param_name].parents['upper'])
                    self.assertIs(param.lower, model.group_nodes[param_name].parents['lower'])

                if model.is_group_model and param.create_subj_nodes and not param.is_bottom_node:
                    print "SUBJ NODES"
                    # Check if subj node exist
                    # TODO: What went wrong here?
                    #self.assertIn(param_name, model.var_nodes.keys())
                    #self.assertIn(param_name+'_var', model.nodes.keys())
                    self.assertIn(param_name, model.subj_nodes.keys())
                    self.assertIn(param_name+'_subj', model.nodes.keys())

                    # Check if subj node parents link to the correct group nodes
                    if param.create_group_node:
                        for i_subj in range(model._num_subjs):
                            self.assertIs(model.group_nodes[param_name], model.subj_nodes[param_name][i_subj].parents['mu'])
                            self.assertIs(model.var_nodes[param_name], model.subj_nodes[param_name][i_subj].parents['tau'])

                if param.is_bottom_node:
                    self.assertIn(param_name, model.bottom_nodes.keys())
                    if model.is_group_model:
                        for i_subj in range(model._num_subjs):
                            parents = model.bottom_nodes[param_name][i_subj].parents.values()
                            # Test if parents are linked to the correct subjs
                            for parent in parents:
                                if not parent == 1:
                                    self.assertIn(parent.__name__, [model.subj_nodes[x][i_subj].__name__ for x in model.subj_nodes.keys()])



class TestHierarchicalBreakDown(unittest.TestCase):
    """
    simple tests to see if hierarchical merthods do not raise and error
    """

    @classmethod
    def setUpClass(self):

        #load models
        self.models = test_utils.load_models()

        #run models
        test_utils.sample_from_models(self.models, n_iter=200)

    def runTest(self):
        pass

    def test_map(self):
        for model in self.models:
            if model.is_group_model:
                try:
                    model.map()
                    raise ValueError("NotimplementedError should have been raised")
                except NotImplementedError:
                    pass
            else:
                model.map(runs=2)
    
    def test_dic_info(self):
        for model in self.models:
            model.dic_info()
            
    def test_print_stats(self):
        for model in self.models:
            model.print_stats()
            model.print_group_stats()
    
    @unittest.skip("Not implemented")    
    def load_db(self):
        pass

    def test_init_from_existing_model(self):
        new_models = test_utils.load_models()
        for (i_m, pre_model) in enumerate(self.models):
            new_models[i_m].init_from_existing_model(pre_model)
        
    def test_plot_posteriors(self):
        pass
    
    def test_subj_by_subj_map_init(self):
        models = test_utils.load_models()
        for model in models:
            if model.is_group_model:
                model.subj_by_subj_map_init(runs=1)



# class TestBayesianANOVA(unittest.TestCase):
#     def __init__(self, *args, **kwargs):
#         return
#         super(TestBayesianANOVA, self).__init__(*args, **kwargs)

#         np.random.seed(31337)

#         params = {}
#         self.data = kabuki.utils.generate_effect_data(2, 1, .5, -1.5, 15, 50)

#         # Generate model
#         self.model = kabuki.ANOVA(data, is_subj_model=True, depends_on={'effect':['cond']})
#         self.model.mcmc(map_=False)

#     def testEstimates(self):
#         pass
