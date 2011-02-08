import kabuki
import numpy as np
import unittest
import pymc as pm


@kabuki.hierarchical
class TestClass(kabuki.Prototype):
    param_names = ('test',)
        
    def get_root_param(self, param, all_params, tag, pos):
        return pm.Uniform('%s%s'% (param,tag), lower=0, upper=1)

    def get_tau_param(self, param, all_params, tag):
        return pm.Uniform('%s%s'% (param,tag), lower=0, upper=10)
    
    def get_subj_param(self, param_name, parent_mean, parent_tau, subj_idx, all_params, tag, pos):
        return pm.Normal('%s%s%i'%(param_name, tag, subj_idx), mu=parent_mean, tau=parent_tau)

    def get_observed(self, name, subj_data, params, idx=None, pos=None):
        return pm.Normal('%s%i' % (name,idx), mu=params['test'][idx], tau=1, value=subj_data['score'], observed=True)
    
class TestHierarchical(unittest.TestCase):
    def runTest(self):
        pass

    def __init__(self, *args, **kwargs):
        super(TestHierarchical, self).__init__(*args, **kwargs)
        
        np.random.seed(31337)
        self.num_subjs = 5
        self.subjs = range(self.num_subjs)
        pts_per_subj = 100
        self.data = np.empty(self.num_subjs*pts_per_subj, dtype=([('subj_idx',np.int), ('score',np.float), ('dep', 'S8')]))

        for subj in self.subjs:
            self.data['subj_idx'][subj*pts_per_subj:(subj+1)*pts_per_subj] = subj
            self.data['score'][subj*pts_per_subj:(subj+1)*pts_per_subj] = subj
            self.data['dep'][subj*pts_per_subj:(subj+1)*pts_per_subj] = 'dep1'
            self.data['dep'][subj*pts_per_subj+pts_per_subj/2:(subj+1)*pts_per_subj] = 'dep2'

    def setUp(self):
        self.test_model = TestClass(data=self.data)
        self.test_model_dep = TestClass(self.data, depends_on={'test':['dep']})

    def test_passing_depends_on(self):
        self.assertEqual(self.test_model_dep.depends_on, {'test':['dep']})
        
    def test_get_data_depend(self):
        self.test_model_dep._set_params()
        
        subj = 0
        dep = ('dep1','dep2')
        i = 0
        data_dep = self.test_model_dep._get_data_depend()
        for (data, param, param_name) in data_dep:
            np.testing.assert_equal(np.unique(data['subj_idx']), self.subjs)
            #self.assertTrue(np.all(data['score'] == subj))
            self.assertTrue(np.all(data['dep'] == dep[i]))

            subj += 1
            i = 1 if i == 0 else 0

    def test_set_params(self):
        self.test_model._set_params()

        self.assertEqual(self.test_model.param_names, ('test',))
        self.assertEqual(self.test_model.group_params.keys(), ['test'])
        np.testing.assert_equal([param.__name__ for param in self.test_model.subj_params['test']], ['test%i'%i for i in self.subjs])

        # Test if parents of subj_params have the correct name
        for parent_params in [param.parents.values() for param in self.test_model.subj_params['test']]:
            np.testing.assert_equal([parent_param.__name__ for parent_param in parent_params], ['test', 'testtau'])

        # Test if parents of subj_params are the correct objects
        for parent_params in self.test_model.subj_params['test']:
            np.testing.assert_equal(parent_params.parents['mu'], self.test_model.group_params['test'])
            np.testing.assert_equal(parent_params.parents['tau'], self.test_model.group_params_tau['test'])
            
    def test_set_params_dep(self):
        self.test_model_dep._set_params()

        self.assertEqual(self.test_model_dep.param_names, ('test',))
        self.assertEqual(self.test_model_dep.group_params.keys(), ["test('dep1',)", "test('dep2',)"])
        np.testing.assert_equal([param.__name__ for param in self.test_model_dep.subj_params["test('dep1',)"]], ["test('dep1',)%i"%i for i in self.subjs])
        np.testing.assert_equal([param.__name__ for param in self.test_model_dep.subj_params["test('dep2',)"]], ["test('dep2',)%i"%i for i in self.subjs])

        # Test if parents of subj_params have the correct name
        for parent_params in [param.parents.values() for param in self.test_model_dep.subj_params["test('dep1',)"]]:
            np.testing.assert_equal([parent_param.__name__ for parent_param in parent_params], ["test('dep1',)", "test('dep1',)tau"])
        # Test if parents of subj_params are the correct objects
        for parent_params in self.test_model_dep.subj_params["test('dep1',)"]:
            np.testing.assert_equal(parent_params.parents['mu'], self.test_model_dep.group_params["test('dep1',)"])
            np.testing.assert_equal(parent_params.parents['tau'], self.test_model_dep.group_params_tau["test('dep1',)"])
            
        # Test if parents of subj_params have the correct name
        for parent_params in [param.parents.values() for param in self.test_model_dep.subj_params["test('dep2',)"]]:
            np.testing.assert_equal([parent_param.__name__ for parent_param in parent_params], ["test('dep2',)", "test('dep2',)tau"])
        # Test if parents of subj_params are the correct objects
        for parent_params in self.test_model_dep.subj_params["test('dep2',)"]:
            np.testing.assert_equal(parent_params.parents['mu'], self.test_model_dep.group_params["test('dep2',)"])
            np.testing.assert_equal(parent_params.parents['tau'], self.test_model_dep.group_params_tau["test('dep2',)"])

    def test_create_observed(self):
        self.test_model_dep._set_params()

        likelihoods = [likelihood for likelihood in self.test_model_dep.likelihoods]
        for i,dep in enumerate(["test('dep1',)", "test('dep2',)"]):
            for j in self.subjs:
                np.testing.assert_equal(self.test_model_dep.likelihoods[i][j].parents['mu'],
                                        self.test_model_dep.subj_params[dep][j])
                np.testing.assert_equal(self.test_model_dep.likelihoods[i][j].parents['mu'],
                                        self.test_model_dep.subj_params[dep][j])

class TestBayesianANOVA(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBayesianANOVA, self).__init__(*args, **kwargs)
        
        np.random.seed(31337)

        params = {'
        self.data = kabuki.utils.generate_effect_data(2, 1, .5, -1.5, 15, 50)

        # Generate model
        self.model = kabuki.ANOVA(data, is_subj_model=True, depends_on={'effect':['cond']})
        self.model.mcmc(map_=False)

    def testEstimates(self):
        
