import kabuki
from kabuki.hierarchical import Parameter
import numpy as np
import unittest
import pymc as pm

class TestClassOneParam(kabuki.Hierarchical):
    params = (Parameter('test', lower=0, upper=1), 
              Parameter('observed', is_bottom_node=True))
        
    def get_group_node(self, param):
        return pm.Uniform(param.full_name, lower=param.lower, upper=param.upper)

    def get_var_node(self, param):
        return pm.Uniform(param.full_name, lower=0, upper=10)
    
    def get_subj_node(self, param):
        return pm.Normal(param.full_name, mu=param.group, tau=param.var)

    def get_bottom_node(self, param, params):
        return pm.Normal(param.full_name, mu=params['test'], var=1, value=param.data['score'], observed=True)

class TestClassTwoParam(kabuki.Hierarchical):
    params = (Parameter('test1', lower=0, upper=1), 
              Parameter('test2', lower=0, upper=1), 
              Parameter('observed', is_bottom_node=True))

    def get_group_node(self, param):
        return pm.Uniform(param.full_name, lower=param.lower, upper=param.upper)

    def get_var_node(self, param):
        return pm.Uniform(param.full_name, lower=0, upper=10)
    
    def get_subj_node(self, param):
        return pm.Normal(param.full_name, mu=param.group, tau=param.var)

    def get_bottom_node(self, param, params):
        a = pm.Normal(param.full_name, mu=params['test'], var=1., value=param.data['score'], observed=True)
        b = pm.Normal(param.full_name, mu=params['test2'], var=1., value=param.data['score'], observed=True)
        return [a,b]
    
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

        for subj in self.subjs:
            self.data['subj_idx'][subj*pts_per_subj:(subj+1)*pts_per_subj] = subj
            self.data['score'][subj*pts_per_subj:(subj+1)*pts_per_subj] = subj
            self.data['dep'][subj*pts_per_subj:(subj+1)*pts_per_subj] = 'dep1'
            self.data['dep'][subj*pts_per_subj+pts_per_subj/2:(subj+1)*pts_per_subj] = 'dep2'
            self.data['foo'][subj*pts_per_subj:(subj+1)*pts_per_subj] = 'bar1'
            self.data['foo'][subj*pts_per_subj+pts_per_subj/2:(subj+1)*pts_per_subj] = 'bar2'

    def setUp(self):
        self.test_model_one_param = TestClassOneParam(self.data)
        self.test_model_one_param_dep = TestClassOneParam(self.data, depends_on={'test':['dep']})

        self.test_model_two_param = TestClassTwoParam(self.data)
        self.test_model_two_param_dep = TestClassTwoParam(self.data, depends_on={'test':['dep']})

        self.test_model_two_param_dep2 = TestClassTwoParam(self.data, depends_on={'test2':['dep']})

        self.test_model_two_param_dep_dual = TestClassTwoParam(self.data, depends_on={'test2':['dep','foo']})

    def test_passing_depends_on(self):
        self.assertEqual(self.test_model_one_param_dep.depends_on, {'test':['dep']})
        self.assertEqual(self.test_model_two_param_dep.depends_on, {'test':['dep']})
        self.assertEqual(self.test_model_two_param_dep2.depends_on, {'test2':['dep']})
        self.assertEqual(self.test_model_two_param_dep_dual.depends_on, {'test2':['dep', 'foo']})
        
    def test_get_data_depend(self):
        self.tst_get_data_depend(self.test_model_one_param_dep)
        self.tst_get_data_depend(self.test_model_two_param_dep)
        self.tst_get_data_depend(self.test_model_two_param_dep2)
        self.tst_get_data_depend(self.test_model_two_param_dep_dual)
        
    def tst_get_data_depend(self, model):
        model.create_nodes()
        
        subj = 0
        dep = ('dep1','dep2')
        i = 0
        data_dep = model._get_data_depend()
        for (data, param, param_name) in data_dep:
            np.testing.assert_equal(np.unique(data['subj_idx']), self.subjs)
            #self.assertTrue(np.all(data['score'] == subj))
            self.assertTrue(np.all(data['dep'] == dep[i]))

            subj += 1
            i = 1 if i == 0 else 0

    def test_create_nodes(self):
        self.test_model_one_param.create_nodes()

        self.assertEqual(set(self.test_model_one_param.params_dict.keys()), set(['test', 'observed']))
        self.assertEqual(self.test_model_one_param.group_params.keys(), ['test'])
        np.testing.assert_equal([param.__name__ for param in self.test_model_one_param.subj_params['test']], ['test%i'%i for i in self.subjs])

        # Test if parents of subj_params have the correct name
        for parent_params in [param.parents.values() for param in self.test_model_one_param.subj_params['test']]:
            np.testing.assert_equal([parent_param.__name__ for parent_param in parent_params], ['test', 'testvar'])

        # Test if parents of subj_params are the correct objects
        for parent_params in self.test_model_one_param.subj_params['test']:
            assert parent_params.parents['mu'] is self.test_model_one_param.group_params['test']
            assert parent_params.parents['var'] is self.test_model_one_param.group_params_var['test']
        
    def test_set_params_dep(self):
        self.tst_set_params_dep(self.test_model_one_param_dep)
        self.tst_set_params_dep_two(self.test_model_two_param_dep)
        self.tst_set_params_dep_two2(self.test_model_two_param_dep2)

    def tst_set_params_dep_two2(self, model):
        model._set_params()

        self.assertEqual(model.param_names, ('test','test2'))
        self.assertEqual(model.group_params.keys(), ["test", "test2('dep1',)", "test2('dep2',)"])
        np.testing.assert_equal([param.__name__ for param in model.subj_params["test2('dep1',)"]], ["test2('dep1',)%i"%i for i in self.subjs])
        np.testing.assert_equal([param.__name__ for param in model.subj_params["test2('dep2',)"]], ["test2('dep2',)%i"%i for i in self.subjs])

        # Test if parents of subj_params have the correct name
        for parent_params in [param.parents.values() for param in model.subj_params["test2('dep1',)"]]:
            np.testing.assert_equal([parent_param.__name__ for parent_param in parent_params], ["test2('dep1',)", "test2('dep1',)var"])
        # Test if parents of subj_params are the correct objects
        for parent_params in model.subj_params["test2('dep1',)"]:
            assert parent_params.parents['mu'] is model.group_params["test2('dep1',)"]
            assert parent_params.parents['var'] is model.group_params_var["test2('dep1',)"]
            
        # Test if parents of subj_params have the correct name
        for parent_params in [param.parents.values() for param in model.subj_params["test2('dep2',)"]]:
            np.testing.assert_equal([parent_param.__name__ for parent_param in parent_params], ["test2('dep2',)", "test2('dep2',)var"])
        # Test if parents of subj_params are the correct objects
        for parent_params in model.subj_params["test2('dep2',)"]:
            assert parent_params.parents['mu'] is model.group_params["test2('dep2',)"]
            assert parent_params.parents['var'] is model.group_params_var["test2('dep2',)"]

    def tst_set_params_dep_two(self, model):
        model._set_params()

        self.assertEqual(set(model.params_dict.keys()), set(['test','test2','observed']))
        self.assertEqual(model.group_params.keys(), ["test('dep1',)", "test('dep2',)", "test2"])
        np.testing.assert_equal([param.__name__ for param in model.subj_params["test('dep1',)"]], ["test('dep1',)%i"%i for i in self.subjs])
        np.testing.assert_equal([param.__name__ for param in model.subj_params["test('dep2',)"]], ["test('dep2',)%i"%i for i in self.subjs])

        # Test if parents of subj_params have the correct name
        for parent_params in [param.parents.values() for param in model.subj_params["test('dep1',)"]]:
            np.testing.assert_equal([parent_param.__name__ for parent_param in parent_params], ["test('dep1',)", "test('dep1',)var"])
        # Test if parents of subj_params are the correct objects
        for parent_params in model.subj_params["test('dep1',)"]:
            assert parent_params.parents['mu'] is model.group_params["test('dep1',)"]
            assert parent_params.parents['var'] is model.group_params_var["test('dep1',)"]
            
        # Test if parents of subj_params have the correct name
        for parent_params in [param.parents.values() for param in model.subj_params["test('dep2',)"]]:
            np.testing.assert_equal([parent_param.__name__ for parent_param in parent_params], ["test('dep2',)", "test('dep2',)var"])
        # Test if parents of subj_params are the correct objects
        for parent_params in model.subj_params["test('dep2',)"]:
            assert parent_params.parents['mu'] is model.group_params["test('dep2',)"]
            assert parent_params.parents['var'] is model.group_params_var["test('dep2',)"]


    def tst_set_params_dep(self, model):
        model.create_nodes()

        self.assertEqual(set(model.params_dict.keys()), set(['test','observed']))
        #self.assertEqual(model.group_nodes.keys(), ["test('dep1',)", "test('dep2',)"])
        #np.testing.assert_equal([param.__name__ for param in model.subj_nodes["test('dep1',)"]], ["test('dep1',)%i"%i for i in self.subjs])
        #np.testing.assert_equal([param.__name__ for param in model.subj_nodes["test('dep2',)"]], ["test('dep2',)%i"%i for i in self.subjs])

        # Test if parents of subj_params have the correct name
        for parent_params in [param.parents.values() for param in model.subj_nodes["test('dep1',)"]]:
            np.testing.assert_equal([parent_param.__name__ for parent_param in parent_params], ["test('dep1',)", "test('dep1',)var"])
        # Test if parents of subj_params are the correct objects
        for parent_params in model.subj_params["test('dep1',)"]:
            assert parent_params.parents['mu'] is model.group_nodes["test('dep1',)"]
            assert parent_params.parents['var'] is model.group_nodes_var["test('dep1',)"]
            
        # Test if parents of subj_params have the correct name
        for parent_params in [param.parents.values() for param in model.subj_nodes["test('dep2',)"]]:
            np.testing.assert_equal([parent_param.__name__ for parent_param in parent_params], ["test('dep2',)", "test('dep2',)var"])
        # Test if parents of subj_params are the correct objects
        for parent_params in model.subj_params["test('dep2',)"]:
            assert parent_params.parents['mu'] is model.group_nodes["test('dep2',)"]
            assert parent_params.parents['var'] is model.group_nodes_var["test('dep2',)"]

    def tst_create_observed(self, model):
        model.create_nodes()

        likelihoods = [likelihood for likelihood in model.likelihoods]
        for i,dep in enumerate(["test('dep1',)", "test('dep2',)"]):
            for j in self.subjs:
                np.testing.assert_equal(model.likelihoods[i][j].parents['mu'],
                                        model.subj_params[dep][j])
                np.testing.assert_equal(model.likelihoods[i][j].parents['mu'],
                                        model.subj_params[dep][j])

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
