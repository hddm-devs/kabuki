import unittest
import kabuki
import numpy as np
import pymc as pm
import math
import scipy as sc
import collections
from pprint import pprint
from numpy.random import randn
from numpy import array, sqrt
from nose import SkipTest
from pandas import DataFrame
from time import time
import unittest

from kabuki.utils import stochastic_from_dist

def multi_normal_like(values, vec_mu, tau):
    """logp for multi normal"""
    logp = 0
    for i in xrange(len(vec_mu)):
        logp += pm.normal_like(values[i,:], vec_mu[i], tau)

    return logp

MN = stochastic_from_dist(name="MultiNormal", logp=multi_normal_like)



class TestStepMethods(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestStepMethods, self).__init__(*args, **kwargs)
        self.uniform_lb = 1e-10
        self.uniform_ub = 1e10

    def runTest(self):
        return


    def assert_results(self, node, true_value, true_mean, true_std=None,
                       mean_tol=0.1, std_tol=0.2):
        """check if the sampler output agree with the analytical meand and
        analytical std
        Input:
            Node - the node to check
            true_value - the true value of the node
            true_mean - the true mean
            true_std - the std of the distribution (None if it's unknown)
            mean_tol - the tolerance to use when checking the difference between
                the true_mean and the sampled mean
            std_tol - same as mean_tol but for checking the std
        """



        pprint(node.stats())
        lb = node.stats()['quantiles'][2.5]
        ub = node.stats()['quantiles'][97.5]
        if not (lb <  true_value < ub):
            print "Warnnig!!!!, sigma was not found in the credible set"


        print "true value:     ", true_value
        print "sampled median: ", node.stats()['quantiles'][50]
        print "sampled mean:   ", node.stats()['mean']
        print "true mean:      ", true_mean
        if true_std is not None:
            print "true std:       ", true_std
            print "sampled std:    ", node.stats()['standard deviation']

        np.testing.assert_allclose(node.stats()['mean'], true_mean, rtol=mean_tol)
        if true_std is not None:
            np.testing.assert_allclose(node.stats()['standard deviation'], true_std, rtol=std_tol)


    def normal_normal(self, add_shift, sigma_0, sigma_beta, sigma_y, true_mu,
                     n_subjs, avg_samples, seed, use_metropolis):
        """check the normal_normal configuration
        Model:
        x ~ N(mu_0, sigma_0**-2)
        y ~ N(x + b, sigma_y**-2)
        only mu is Stochastic

        b is constant in the model. it is generated from N(o,sigma_b**-2)
        y is generated from N(true_mu + b, sigma_y**-2)
            add_shift - whether to add some

        n_subjs - number of b
        avg_samples - the average samples per subject

        """

        np.random.seed(seed)

        #create nodes
        tau_0 = sigma_0**-2
        mu_0 = 0.
        nodes, size, x_values = \
        self.create_nodes_for_normal_normal(add_shift, tau_0, mu_0, sigma_beta,
                                       sigma_y, true_mu, n_subjs, avg_samples)
        mu = nodes['mu']

        #sample
        mm = pm.MCMC(nodes)
        if use_metropolis:
            mm.sample(20000,5000)
        else:
            mm.use_step_method(kabuki.steps.kNormalNormal,mu)#, b=b)
            mm.sample(10000)

        #calc the new distrbution
        total_n = sum(size)
        tau = sigma_y**-2
        sum_obs = sum([sum(x.value.flatten()) for x in mm.observed_stochastics])
        if add_shift:
            tmp = sum(array(size)* x_values)
        else:
            tmp = 0

        tau_prime = tau_0 + total_n*tau
        mu_prime = (tau*(sum_obs - tmp) + mu_0*tau_0)/tau_prime
        true_std = 1./np.sqrt(tau_prime)

        self.assert_results(mu, true_mu, mu_prime, true_std, mean_tol=0.1, std_tol=0.1)

        return mm, mu_prime, true_std

    def create_nodes_for_normal_normal(self, add_shift, tau_0, mu_0, sigma_beta,
                                       sigma_y, true_mu, n_subjs, avg_samples):
        """ create the normal normal nodes"""

        mu = pm.Normal('mu',mu_0,tau_0)
        nodes = {'mu': mu}
        size = [None]*n_subjs
        x_values = [None]*n_subjs
        if add_shift:
            b = []
        else:
            b = None
        for i in range(n_subjs):
            size[i] = int(max(1, avg_samples + randn()*10))
            if add_shift:
                x_values[i] = randn()*sigma_beta
                value = randn(size[i]) * sigma_y + true_mu + x_values[i]
                x = pm.Lambda('x%d' % i, lambda x=x_values[i]:x)
                y = pm.Normal('y%d' % i,mu+x, sigma_y**-2, value=value,observed=True)
                nodes['x%d' % i] = x
                b.append(x)
            else:
                value = randn(size[i]) * sigma_y + true_mu
                y = pm.Normal('y%d' % i,mu, sigma_y**-2, value=value,observed=True)

            nodes['y%d' % i] = y

        return nodes, size, x_values


    def normal_normal_bundle(self, use_metropolis):
        """run normal_normal with different parameters"""
        self.normal_normal(add_shift=True, sigma_0=100., sigma_beta=2., sigma_y=1.5,
                         true_mu=-3., n_subjs=1, avg_samples=100, seed=1, use_metropolis=use_metropolis)
        self.normal_normal(add_shift=True, sigma_0=50., sigma_beta=3., sigma_y=2,
                         true_mu=-2., n_subjs=2, avg_samples=10, seed=2, use_metropolis=use_metropolis)
        self.normal_normal(add_shift=True, sigma_0=10., sigma_beta=1., sigma_y=2.5,
                         true_mu=-1., n_subjs=3, avg_samples=10, seed=3, use_metropolis=use_metropolis)
        self.normal_normal(add_shift=False, sigma_0=1., sigma_beta=0.5, sigma_y=0.5,
                         true_mu=-4., n_subjs=4, avg_samples=50, seed=4, use_metropolis=use_metropolis)
        self.normal_normal(add_shift=False, sigma_0=50., sigma_beta=0.3, sigma_y=1.5,
                         true_mu=-6., n_subjs=5, avg_samples=50, seed=5, use_metropolis=use_metropolis)
        self.normal_normal(add_shift=False, sigma_0=100., sigma_beta=0.75, sigma_y=2.5,
                         true_mu=100., n_subjs=6, avg_samples=30, seed=6, use_metropolis=use_metropolis)

    def test_normal_normal_solution(self):
        """test normal normal analytic solution"""
        self.normal_normal_bundle(use_metropolis=True)


    def test_kNormalNormal(self):
        """test normal_normal step method"""
        self.normal_normal_bundle(use_metropolis=False)


    def create_nodes_for_PriorNormalstd(self, n_subjs, sigma_0, mu_0, prior, **kwargs):
        """"create node for models with PriorNormalstd step method"""
        #create nodes
        if prior is pm.Uniform:
            sigma = pm.Uniform('sigma', self.uniform_lb, self.uniform_ub, value=1.)
        elif prior is kabuki.utils.HalfCauchy:
            sigma = kabuki.utils.HalfCauchy('sigma', **kwargs)

        x_values = []
        nodes = {'sigma': sigma}
        if not isinstance(mu_0, collections.Sequence):
            mu_0 = [mu_0]
        n_conds = len(mu_0)

        for i_cond in range(n_conds):
            for i in range(n_subjs):
                x_value = randn()*sigma_0 + mu_0[i_cond]
                x_values.append(x_value)
                x = pm.Normal('x%d' % i, mu_0[i_cond], sigma**-2, value=x_value, observed=True)
                nodes['x%d_%d' % (i, i_cond)] = x

        return nodes, x_values


    def uniform_normalstd(self, sigma_0, mu_0, n_subjs, seed, use_metropolis):
        """test estimation of Normal distribution std with uniform prior
            sigma_0 - the value of the std noe
            mu_0 - the value of the mu node
            use_metropolis - should it use metropolis to evaluate the sampled mean
                instead of the UniformPriorNormalstd
        """

        np.random.seed(seed)

        nodes, x_values = self.create_nodes_for_PriorNormalstd(n_subjs, sigma_0, mu_0, prior=pm.Uniform)
        sigma = nodes['sigma']
        mm = pm.MCMC(nodes)

        if use_metropolis:
            mm.sample(20000,5000)
        else:
            mm.use_step_method(kabuki.steps.UniformPriorNormalstd, sigma)
            mm.sample(10000)



        #calc the new distrbution
        alpha = (n_subjs - 1) / 2.
        beta  = sum([(x - mu_0)**2 for x in x_values]) / 2.
        true_mean = math.gamma(alpha-0.5)/math.gamma(alpha)*np.sqrt(beta)
        anal_var = beta / (alpha - 1) - true_mean**2
        true_std = np.sqrt(anal_var)

        self.assert_results(sigma, sigma_0, true_mean, true_std)
        return mm

    def uniform_normalstd_multiple_conds_with_shared_sigma(self, sigma_0, mu_0, n_subjs, seed, use_metropolis):
        """test estimation of Normal distribution std with uniform prior
            sigma_0 - the value of the std noe
            mu_0 - the value of the mu node
            use_metropolis - should it use metropolis to evaluate the sampled mean
                instead of the UniformPriorNormalstd
        """

        np.random.seed(seed)
        n_conds = len(mu_0)

        nodes, x_values = self.create_nodes_for_PriorNormalstd(n_subjs, sigma_0, mu_0, prior=pm.Uniform)
        sigma = nodes['sigma']
        mm = pm.MCMC(nodes)

        if use_metropolis:
            mm.sample(20000,5000)
        else:
            mm.use_step_method(kabuki.steps.UniformPriorNormalstd, sigma)
            mm.sample(10000)



        #calc the new distrbution
        alpha = (n_subjs*n_conds - 1) / 2.
        beta = 0
        for i_cond in range(n_conds):
            cur_x_values = x_values[i_cond*n_subjs:(i_cond+1)*n_subjs]
            beta  += sum([(x - mu_0[i_cond])**2 for x in cur_x_values]) / 2.
        true_mean = math.gamma(alpha-0.5)/math.gamma(alpha)*np.sqrt(beta)
        anal_var = beta / (alpha - 1) - true_mean**2
        true_std = np.sqrt(anal_var)

        self.assert_results(sigma, sigma_0, true_mean, true_std)
        return mm


    def test_uniform_normalstd_numerical_solution(self):
        """test uniform_normalstd with Metropolis to evaluate the numerical solution of
        the mean and std"""
        self.uniform_normalstd(sigma_0=0.5, mu_0=0, n_subjs=8, seed=1, use_metropolis=True)
        self.uniform_normalstd(sigma_0=1.5, mu_0=-100, n_subjs=4, seed=2, use_metropolis=True)
        self.uniform_normalstd(sigma_0=2.5, mu_0=2, n_subjs=5, seed=3, use_metropolis=True)
        self.uniform_normalstd(sigma_0=3.5, mu_0=-4, n_subjs=7, seed=4, use_metropolis=True)
#        self.uniform_normalstd(sigma_0=4.5, mu_0=10, n_subjs=4, seed=5, use_metropolis=True)

    def test_UniformNormalstd_step_method(self):
        """test UniformPriorNormalstd step method"""
        self.uniform_normalstd(sigma_0=0.5, mu_0=0, n_subjs=8, seed=1, use_metropolis=False)
        self.uniform_normalstd(sigma_0=1.5, mu_0=-100, n_subjs=4, seed=2, use_metropolis=False)
        self.uniform_normalstd(sigma_0=2.5, mu_0=2, n_subjs=5, seed=3, use_metropolis=False)
        self.uniform_normalstd(sigma_0=3.5, mu_0=-4, n_subjs=7, seed=4, use_metropolis=False)
        self.uniform_normalstd(sigma_0=4.5, mu_0=10, n_subjs=4, seed=5, use_metropolis=False)

    def test_uniform_normalstd_with_multiple_condition_numerical_solution(self):
        """test uniform_normalstd with Metropolis to evaluate the numerical solution of
        the mean and std"""
        self.uniform_normalstd_multiple_conds_with_shared_sigma(sigma_0=0.5, mu_0=(0,2,10), n_subjs=8, seed=1, use_metropolis=True)
        self.uniform_normalstd_multiple_conds_with_shared_sigma(sigma_0=3.5, mu_0=(-100,3), n_subjs=4, seed=2, use_metropolis=True)
        self.uniform_normalstd_multiple_conds_with_shared_sigma(sigma_0=2.5, mu_0=(1,2), n_subjs=5, seed=3, use_metropolis=True)
        self.uniform_normalstd_multiple_conds_with_shared_sigma(sigma_0=0.5, mu_0=(-4,-3,2,1,0), n_subjs=7, seed=4, use_metropolis=True)

    def test_UniformNormalstd_step_method_with_multiple_condition(self):
        """test UniformPriorNormalstd step method"""
        self.uniform_normalstd_multiple_conds_with_shared_sigma(sigma_0=0.5, mu_0=(0,2,10), n_subjs=8, seed=1, use_metropolis=False)
        self.uniform_normalstd_multiple_conds_with_shared_sigma(sigma_0=3.5, mu_0=(-100,3), n_subjs=4, seed=2, use_metropolis=False)
        self.uniform_normalstd_multiple_conds_with_shared_sigma(sigma_0=2.5, mu_0=(1,2), n_subjs=5, seed=3, use_metropolis=False)
        self.uniform_normalstd_multiple_conds_with_shared_sigma(sigma_0=0.5, mu_0=[-4], n_subjs=7, seed=4, use_metropolis=False)



    def numerical_solution(self, defective_posterior, lb, ub):
        """numerical estimation of the mean and std from defective posterior
        defective_posterior <func> - the defective posterior
        lb - lower bound
        ub - upper bound
        """

        norm_factor = sc.integrate.quad(defective_posterior,lb,ub)[0]
        #function to compute moments
        moment = lambda x,n=1: defective_posterior(x) * (x**n) / norm_factor

        #computing mean and std
        true_mean = sc.integrate.quad(moment,lb, ub, args=(1))[0]
        m2 = sc.integrate.quad(moment,lb, ub, args=(2))[0]
        anal_var = m2 - true_mean**2
        true_std = np.sqrt(anal_var)
        return true_mean, true_std


    def half_cauchy_normal_std(self, sigma_0=1., mu_0=0., S=10, n_subjs=8, seed=1,
                            use_metropolis=False):
        """test estimation of Normal distribution std with halh Cauchy prior
            sigma_0 - the value of the std noe
            mu_0 - the value of the mu node
            S - the scale of the half Cauchy
            use_metropolis - should it use metropolis to evaluate the sampled mean
                instead of the UniformPriorNormalstd
        """


        #create model
        np.random.seed(seed)
        nodes, x_values = \
        self.create_nodes_for_PriorNormalstd(n_subjs, sigma_0, mu_0,
                                             prior=kabuki.utils.HalfCauchy,
                                             S=S, value=1)
        sigma = nodes['sigma']

        #sample
        mm = pm.MCMC(nodes)
        if use_metropolis:
            mm.sample(20000,5000)
        else:
            mm.use_step_method(kabuki.steps.HCauchyPriorNormalstd, sigma)
            mm.sample(10)

        #compute defective posterior
        beta = sum((array(x_values) - mu_0)**2)/2
        def defective_posterior(x, n=n_subjs, beta=beta, S=S):
           gammapdf = (x**2)**(-n/2.) * np.exp(-beta/(x**2))
           cauchy = S / (x**2 + S**2)
           return gammapdf * cauchy

        #check results
        true_mean, true_std = self.numerical_solution(defective_posterior, 0, np.inf)
        self.assert_results(sigma, sigma_0, true_mean, true_std)
        return mm

    def half_cauchy_bundle(self, use_metropolis):
        self.half_cauchy_normal_std(sigma_0=4.5, mu_0=0, n_subjs=8, seed=1, S=5,
                                    use_metropolis=use_metropolis)
        self.half_cauchy_normal_std(sigma_0=0.5, mu_0=-100, n_subjs=4, seed=2, S=20,
                                    use_metropolis=use_metropolis)
        self.half_cauchy_normal_std(sigma_0=5.5, mu_0=2, n_subjs=5, seed=3, S=3,
                                    use_metropolis=use_metropolis)
        self.half_cauchy_normal_std(sigma_0=1.5, mu_0=-4, n_subjs=7, seed=4, S=10,
                                    use_metropolis=use_metropolis)
        self.half_cauchy_normal_std(sigma_0=4.5, mu_0=10, n_subjs=4, seed=5, S=15,
                                    use_metropolis=use_metropolis)

    def test_half_cauchy_numerical_solution(self):
        """test half_cauchy_normal_std with Metropolis to evaluate the numerical solution of
        the mean and std"""
        self.half_cauchy_bundle(use_metropolis=True)

    def test_HCauchyNormalstd_step_method(self):
        """test HCauchy step method"""
        raise SkipTest("The HCauchy gibbs step method does not work.")
        self.half_cauchy_bundle(use_metropolis=False)

    @unittest.skip("Takes forever to complete.")
    def run_SPXcentered(self, sigma_x, n_subjs, size, mu_value, mu_step_method, seed):
        """ run a single Spxcentered test"""

        #init basic  mcmc
        if np.isscalar(mu_value):
            n_conds = 1
        else:
            n_conds = len(mu_value)

        max_tries = 5
        iter = 10000 #100000
        burnin= 5000 #90000
        nodes, t_values = self.create_hierarchical_model(sigma_x=sigma_x, n_subjs=n_subjs, size=size,
                                                         mu_value=mu_value, seed=seed)
        mcmc = pm.MCMC(nodes)
        [mcmc.use_step_method(mu_step_method, node) for node in nodes['mu']]

        #init mcmc with SPX step method
        nodes_spx, t_values = self.create_hierarchical_model(sigma_x=sigma_x, n_subjs=n_subjs, size=size,
                                                             mu_value=mu_value, seed=seed)
        mcmc_spx = pm.MCMC(nodes_spx)
        mcmc_spx.use_step_method(kabuki.steps.SPXcentered, loc=nodes_spx['mu'],
                                 scale=nodes_spx['sigma'],
                                 loc_step_method=mu_step_method)


        #init mcmc with spx on vec model
        nodes_vpx, t_values = self.create_hierarchical_model(sigma_x=sigma_x, n_subjs=n_subjs, size=size,
                                                             mu_value=mu_value, seed=seed, vec=True)
        mcmc_vpx = pm.MCMC(nodes_vpx)
        mcmc_vpx.use_step_method(kabuki.steps.SPXcentered, loc=nodes_vpx['mu'],
                                 scale=nodes_vpx['sigma'],
                                 loc_step_method=mu_step_method)


        #run all the models until they converge to the same values
        i_try = 0
        while i_try < max_tries:
            print "~~~~~ trying for the %d time ~~~~~~" % (i_try + 1)

            #run spx mcmc
            i_t = time()
            mcmc_spx.sample(iter,burnin)
            print "spx sampling took %.2f seconds" % (time() - i_t)
            stats = dict([('mu%d spx' %x, mcmc_spx.mu[x].stats()) for x in range(n_conds)])

            #run vpx mcmc
            i_t = time()
            mcmc_vpx.sample(iter,burnin)
            print "vpx sampling took %.2f seconds" % (time() - i_t)
            stats.update(dict([('mu%d vpx' %x, mcmc_vpx.mu[x].stats()) for x in range(n_conds)]))

            #run basic mcmc
            i_t = time()
            mcmc.sample(iter,burnin)
            print "basic sampling took %.2f seconds" % (time() - i_t)
            stats.update(dict([('mu%d basic' %x, mcmc.mu[x].stats()) for x in range(n_conds)]))

            df = DataFrame(stats, index=['mean', 'standard deviation']).T
            df = df.rename(columns = {'mean':'mean', 'standard deviation': 'std'})
            print df

            #check if all the results are close enough
            try:
                for i in range(len(df)/3):
                    np.testing.assert_allclose(df[(3*i+0):(3*i+1)], df[(3*i+1):(3*i+2)], atol=0.1, rtol=0.01)
                    np.testing.assert_allclose(df[(3*i+1):(3*i+2)], df[(3*i+2):(3*i+3)], atol=0.1, rtol=0.01)
                    np.testing.assert_allclose(df[(3*i+2):(3*i+3)], df[(3*i+0):(3*i+1)], atol=0.1, rtol=0.01)

                break
            #if not add more runs
            except AssertionError:
                print "Failed to reach agreement. trying again"
                i_try += 1

        assert (i_try < max_tries), "could not replicate values using different mcmc samplers"


    @unittest.skip("Takes forever to complete.")
    def test_SPX(self):
        """test a bundle of SPXcentered tests"""
        print "*************** Test 1 ***************"
        self.run_SPXcentered(sigma_x=1, n_subjs=5, size=100, mu_value=4,
                             mu_step_method=kabuki.steps.kNormalNormal, seed=1)
        print "*************** Test 2 ***************"
        self.run_SPXcentered(sigma_x=1, n_subjs=5, size=10, mu_value=(4,3,2,1,0,4,3,2,1,0),
                             mu_step_method=kabuki.steps.kNormalNormal, seed=1)
        print "*************** Test 3 ***************"
        self.run_SPXcentered(sigma_x=0.5, n_subjs=5, size=10, mu_value=(4,3),
                             mu_step_method=kabuki.steps.kNormalNormal, seed=1)
        print "*************** Test 4 ***************"
        self.run_SPXcentered(sigma_x=0.1, n_subjs=5, size=10, mu_value=(4,3),
                             mu_step_method=kabuki.steps.kNormalNormal, seed=1)
        print "*************** Test 5 ***************"
        self.run_SPXcentered(sigma_x=1, n_subjs=5, size=10, mu_value=range(20),
                             mu_step_method=kabuki.steps.kNormalNormal, seed=1)
        print "*************** Test 6 ***************"
        self.run_SPXcentered(sigma_x=0.1, n_subjs=5, size=10, mu_value=range(20),
                             mu_step_method=kabuki.steps.kNormalNormal, seed=1)

    def create_hierarchical_model(self, sigma_x=1, n_subjs=5, size=100, mu_value=4, seed=1, vec=False):
        """
        create an hierarchical normal model
        y_ijk ~ N(x_jk, 1) (i sample, j subject, k condition)
        x_jk ~ N(m_k, sigma_x**2)
        m_k ~ N(0, 100*-2)

        Input:
            m_value <list> - m_value[k] - m_k
            size <int>- number of samples per subject per category
            n_subjs <int> - no. of subjects
            vec <boolean> - use a vectorized model
        """
        #init
        np.random.seed(seed)
        if np.isscalar(mu_value):
            mu_value = [mu_value]
        n_conds = len(mu_value)
        mu = [None] * n_conds
        subj_nodes = [None]*n_conds
        data_nodes = [None]*n_conds

        #true_values
        true_values = {}
        true_values['sigma'] = sigma_x
        true_values['mu'] = mu_value

        #init sigma node
        sigma = pm.Uniform('sigma', 1e-10,1e10, value=1)
        tau = sigma**-2

        #create nodes for each cond
        for i_cond in range(n_conds):
            #initalize the true value of x
            true_x = randn(n_subjs)*sigma_x + mu_value[i_cond]
            value = np.random.randn(n_subjs, size).T + true_x
            value = value.T
            print true_x

            #init mu and sigma
            mu[i_cond] = pm.Normal('mu%d' % i_cond, 0, 100.**-2, value=0)

            #create subj_nodes (x + y)
            if vec:
                subj_nodes[i_cond] = pm.Normal('x%d' % (i_cond), mu[i_cond], tau, size=n_subjs)
                data_nodes[i_cond] = MN('y%d' % (i_cond), vec_mu=subj_nodes[i_cond], tau=1, value=value, observed=True)
            else:
                subj_nodes[i_cond] = [None]*n_subjs
                data_nodes[i_cond] = [None]*n_subjs
                for i_subj in range(n_subjs):
                    #x is generate from the mean.
                    subj_nodes[i_cond][i_subj] = pm.Normal('x%d_%d' % (i_cond, i_subj), mu[i_cond], tau)
                    data_nodes[i_cond][i_subj] = pm.Normal('y%d_%d' % (i_cond, i_subj),
                                                           mu=subj_nodes[i_cond][i_subj],
                                                           tau=1, value=value[i_subj,:], observed=True)

        #create nodes dictionary
        nodes = {}
        nodes['x'] = subj_nodes
        nodes['y'] = data_nodes
        nodes['mu'] = mu
        nodes['sigma'] = sigma

        return nodes, true_values


    def run_SliceStep(self, sigma_x, n_subjs, size, mu_value, seed, left=None, max_tries=5):

        #init basic  mcmc
        if np.isscalar(mu_value):
            n_conds = 1
        else:
            n_conds = len(mu_value)

        iter = 10000 #100000
        burnin= 5000 #90000

        #init basic mcmc
        nodes, t_values = self.create_hierarchical_model(sigma_x=sigma_x, n_subjs=n_subjs, size=size,
                                                         mu_value=mu_value, seed=seed)
        mcmc = pm.MCMC(nodes)
        [mcmc.use_step_method(kabuki.steps.kNormalNormal, node) for node in nodes['mu']]

        #init mcmc with slice step
        nodes_s, t_values = self.create_hierarchical_model(sigma_x=sigma_x, n_subjs=n_subjs, size=size,
                                                         mu_value=mu_value, seed=seed)
        mcmc_s = pm.MCMC(nodes_s)
        [mcmc_s.use_step_method(kabuki.steps.kNormalNormal, node) for node in nodes_s['mu']]
        mcmc_s.use_step_method(kabuki.steps.SliceStep, nodes_s['sigma'], width=3, left=left)


        #run all the models until they converge to the same values
        i_try = 0
        stats = {}
        while i_try < max_tries:
            print "~~~~~ trying for the %d time ~~~~~~" % (i_try + 1)

            #run slice mcmc
            i_t = time()
            mcmc_s.sample(iter,burnin)
            print "slice sampling took %.2f seconds" % (time() - i_t)
            stats.update(dict([('mu%d S' %x, mcmc_s.mu[x].stats()) for x in range(n_conds)]))

            #run basic mcmc
            i_t = time()
            mcmc.sample(iter,burnin)
            print "basic sampling took %.2f seconds" % (time() - i_t)
            stats.update(dict([('mu%d basic' %x, mcmc.mu[x].stats()) for x in range(n_conds)]))

            df = DataFrame(stats, index=['mean', 'standard deviation']).T
            df = df.rename(columns = {'mean':'mean', 'standard deviation': 'std'})
            print df

            #check if all the results are close enough
            try:
                for i in range(len(df)/2):
                    np.testing.assert_allclose(df[(2*i+0):(2*i+1)], df[(2*i+1):(2*i+2)], atol=0.1, rtol=0.01)
                break
            #if not add more runs
            except AssertionError:
                print "Failed to reach agreement In:"
                print df[(2*i):(2*(i+1))]
                print "trying again"

            i_try += 1

        assert (i_try < max_tries), "could not replicate values using different mcmc samplers"

        return mcmc, mcmc_s

    def test_SliceStep(self):
        """test a bundle of SPXcentered tests"""
        print "*************** Test 1 ***************"
        self.run_SliceStep(sigma_x=1, n_subjs=5, size=100, mu_value=4, seed=1)
        print "*************** Test 2 ***************"
        self.run_SliceStep(sigma_x=1, n_subjs=5, size=10, mu_value=range(10), seed=1)
        # Very slow, causes travis to choke.
        # print "*************** Test 3 ***************"
        # self.run_SliceStep(sigma_x=0.5, n_subjs=5, size=10, mu_value=(4,3), seed=1)
        # print "*************** Test 4 ***************"
        # self.run_SliceStep(sigma_x=0.1, n_subjs=5, size=10, mu_value=(4,3), seed=1)
        # print "*************** Test 5 ***************"
        # self.run_SliceStep(sigma_x=1, n_subjs=5, size=10, mu_value=range(20), seed=1)
        # print "*************** Test 6 ***************"
        # self.run_SliceStep(sigma_x=0.1, n_subjs=5, size=10, mu_value=range(20), seed=1)
        # print "*************** Test 7 ***************"
        # self.run_SliceStep(sigma_x=0.1, n_subjs=5, size=10, mu_value=(4,3), seed=1, left=0)
        # print "*************** Test 8 ***************"
        # self.run_SliceStep(sigma_x=1, n_subjs=5, size=10, mu_value=range(20), seed=1, left=0)
        # print "*************** Test 9 ***************"
        # self.run_SliceStep(sigma_x=0.1, n_subjs=5, size=10, mu_value=range(20), seed=1, left=0)
