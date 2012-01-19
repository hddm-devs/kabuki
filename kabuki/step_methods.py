import pymc as pm
import numpy as np
import kabuki

class NormalPriorNormal(pm.Gibbs):
    """
    Step method for Normal Prior with Normal likelihood.
    """
    child_class = pm.Normal
    parent_label = 'mu'
    target_class = pm.Normal

    def __init__(self, stochastic, *args, **kwargs):
        self.stochastic = stochastic
        self.mu_0 = stochastic.parents['mu']
        self.tau_0 = stochastic.parents['tau']
        self.tau_node = list(stochastic.extended_children)[0].parents['tau']
        self.children = stochastic.children
        self.n_subj = len(self.children)

        pm.Gibbs.__init__(self, stochastic, *args, **kwargs)


    def step(self):

        tau_prime = self.tau_0 + self.n_subj*self.tau_node.value
        sum_v = np.sum([x.value for x in self.children])
        mu_prime = ((self.tau_0 * self.mu_0) + (self.tau_node.value*sum_v))/tau_prime

        self.stochastic.value = np.random.randn()/tau_prime + mu_prime

class PriorNormalstd(pm.Gibbs):
    """
    Step method for Uniform Prior over standard devision of Normal distribution
    using reject sampling
    """

    child_class = pm.Normal
    parent_label = 'tau'

    def __init__(self, stochastic, maxiter=100, **kwargs):
        pm.Gibbs.__init__(self, stochastic, **kwargs)
        self.maxiter = maxiter
        self.alpha = len(self.children) / 2.
        self.mu_node = list(self.children)[0].parents['mu']
        self.fail = 0

    def step(self):

        self.beta  = sum([(x.value - self.mu_node.value)**2 for x in self.children]) / 2.
        reject = True
        iter = 0
        while (reject) and (iter < self.maxiter):
            #sample tau using gamma distrbution
            tau_proposal = np.random.gamma(self.alpha, 1./self.beta)
            #convert to std
            proposal = 1./ np.sqrt(tau_proposal)
            if not self.is_rejected(proposal):
                reject = False
            iter += 1

        if (iter == self.maxiter):
            self.fail += 1
        else:
            self.stochastic.value = proposal

class UniformPriorNormalstd(PriorNormalstd):
    """
    Step method for Uniform Prior over standard devision of Normal distribution
    using reject sampling
    """

    target_class = pm.Uniform

    def __init__(self, stochastic, **kwargs):
        PriorNormalstd.__init__(self, stochastic, **kwargs)
        self.lower = stochastic.parents['lower']
        self.upper = stochastic.parents['upper']


    def is_rejected(self, proposal):
        if (self.lower < proposal < self.upper):
            reject = False
        else:
            reject = True
        return reject


class HCauchyPriorNormalstd(PriorNormalstd):
    """
    Step method for Uniform Prior over standard devision of Normal distribution
    using reject sampling
    """

    target_class = kabuki.utils.HalfCauchy

    def __init__(self, stochastic, **kwargs):
        PriorNormalstd.__init__(self, stochastic, **kwargs)
        self.S = stochastic.parents['S']
        self.logM = np.log(2./(np.pi*self.S))
        self.half_cauchy_logp = kabuki.utils.half_cauchy_logp

    def is_rejected(self, proposal):
        if np.log(np.random.rand()) < (self.half_cauchy_logp(proposal, self.S) - self.logM):
            reject = False
        else:
            reject = True
        return reject