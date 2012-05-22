import kabuki
from kabuki.hierarchical import Knode
import numpy as np
import unittest
import pymc as pm
import test_utils
import scipy
import pandas as pd
from utils import HNodeSimple, HNodeSimpleVar, normal_like

class TestModels(unittest.TestCase):

    #@classmethod
    def setUp(self):
        self.n_subj = 3
        data = kabuki.generate.gen_rand_data(normal_like, {'A':{'loc':0, 'scale':1}, 'B': {'loc':0, 'scale':1}}, subjs=self.n_subj)[0]
        data = pd.DataFrame(data)
        data['condition2'] = np.random.randint(2, size=len(data))
        self.data = data

    def runTest(self):
        pass

    def test_simple_no_deps(self):
        m = HNodeSimple(self.data)
        n_nodes = 1 + self.n_subj*2 #v_g + n_subj * (v_subj + like)
        self.assertEqual(len(m.nodes_db), n_nodes)

    def test_simple_deps(self):
        m = HNodeSimple(self.data, depends_on={'v': 'condition'})
        n_nodes = 2 * (1 + self.n_subj*2) #n_conds * (v_g + n_subj * (v_subj + like))
        self.assertEqual(len(m.nodes_db), n_nodes)

    def test_simplevar_partly_deps(self):
        m = HNodeSimpleVar(self.data, depends_on={'v': 'condition'})
        n_nodes = 1 + 1 + 2 * (1 + self.n_subj*2) #v_std + v_tau + n_conds * (v_g + n_subj * (v_subj + like))
        self.assertEqual(len(m.nodes_db), n_nodes)

    def test_simplevar_deps(self):
        m = HNodeSimpleVar(self.data, depends_on={'v': 'condition', 'v_std':'condition'})
        #n_nodes = n_conds * (v_std + v_tau + v_g + n_subj * (v_subj + like))
        n_nodes = 2 * (1 + 1 + 1 + self.n_subj*2)
        self.assertEqual(len(m.nodes_db), n_nodes)

    def test_simplevar_double_deps_A(self):
        m = HNodeSimpleVar(self.data, depends_on={'v': 'condition', 'v_std':'condition2'})
        #n_nodes = 2*v_tau + 2*v_std + 2*v_g + 4*n_subj*(v_subj + like))
        n_nodes = 2 + 2 + 2 + 4 * self.n_subj * 2
        self.assertEqual(len(m.nodes_db), n_nodes)

    def test_simplevar_double_deps_B(self):
        m = HNodeSimpleVar(self.data, depends_on={'v': ['condition', 'condition2'], 'v_std':'condition2'})
        #n_nodes = 2*v_tau + 2*v_std + 4*v_g + 4*n_subj*(v_subj + like))
        n_nodes = 2 + 2 + 4 + 4 * self.n_subj * 2
        self.assertEqual(len(m.nodes_db), n_nodes)

