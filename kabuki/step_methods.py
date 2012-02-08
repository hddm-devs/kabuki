import pymc as pm
import numpy as np
import kabuki
from numpy.random import randn
from numpy import array, sqrt


class kNormalNormal(pm.Gibbs):
    """
    Step method for Normal Prior with Normal likelihood.
    x ~ N(mu, tau_x)
    y_i ~ N(x + b_i, tau_y)
    """
    child_class = pm.Normal
    parent_label = 'mu'
    target_class = pm.Normal

    def __init__(self, stochastic, *args, **kwargs):

        pm.Gibbs.__init__(self, stochastic, *args, **kwargs)
        assert (self.stochastic != ()), "stochastic should not be a vector"
        self.stochastic = stochastic
        self.mu_0 = stochastic.parents['mu']
        self.tau_0 = stochastic.parents['tau']
        self.tau_node = list(stochastic.extended_children)[0].parents['tau']

        #total number of samples
        self.total_n = sum(array([len(x.value.flatten()) for x in self.children]))

        self.b = [] #holds the b parameters
        self.n_of_b = [] #holds the number of children of each b
        self.shift = False #is the model centered or shifted
        for child in self.children:
            parent = child.parents['mu']
            if parent is stochastic:
                continue
            else:
                self.shift = True
                self.b.append(parent - stochastic)

                self.n_of_b.append(len(child.value.flatten()))

        self.total_b = len(self.b)

    def step(self):

        #get value of mu_0, tau_node and tau_0
        if isinstance(self.mu_0, pm.Node):
            mu_0_val = self.mu_0.value
        else:
            mu_0_val = self.mu_0

        if isinstance(self.tau_node, pm.Node):
            tau_node = self.tau_node.value
        else:
            tau_node = self.tau_node

        if isinstance(self.tau_0, pm.Node):
            tau_0 = self.tau_0.value
        else:
            tau_0 = self.tau_0

        #compute mu_prime and tau_prime
        tau_prime = tau_0 + self.total_n*tau_node
        sum_child_values = np.sum([np.sum(x.value.flatten()) for x in self.children])
        if self.shift:
            xxx = sum([self.n_of_b[i] * self.b[i].value for i in range(self.total_b)])
            temp_mu =  tau_node * (sum_child_values   - xxx)
        else:
            temp_mu = tau_node*sum_child_values
        mu_prime = ((tau_0 * mu_0_val) + temp_mu) / tau_prime

        #sample
        self.stochastic.value = np.random.randn()/np.sqrt(tau_prime) + mu_prime

class PriorNormalstd(pm.Gibbs):
    """
    Step method for Uniform Prior over standard devision of Normal distribution
    using reject sampling
    f(sigma) = f(var)*|2*sigma| = InverseGamma(alpha, beta) = 1./ Gamma(alpha, beta)
    alpha=(n-1)/2,   beta=sum(x_i-mu)**2/2
    so we sample the an r.v. from Gamma and take its inverse, whichgives us var,
    and then we tak the square root of it
    """

    child_class = pm.Normal
    parent_label = 'tau'

    def __init__(self, stochastic, maxiter=100, **kwargs):
        pm.Gibbs.__init__(self, stochastic, **kwargs)
        self.maxiter = maxiter
        n = sum([len(x.value.flatten()) for x in self.children])
        self.alpha = (n - 1) / 2.
        self.mu_node = list(self.children)[0].parents['mu']
        self.fail = 0

    def step(self):

        if isinstance(self.mu_node, pm.Node):
            mu_val = self.mu_node.value
        else:
            mu_val = self.mu_node
        self.beta  = sum([sum((x.value.flatten() - mu_val)**2) for x in self.children]) / 2.
        reject = True
        iter = 0
        while (reject) and (iter < self.maxiter):
            #sample tau using gamma distrbution
            var_proposal = 1./ np.random.gamma(self.alpha, 1./self.beta)
            proposal = np.sqrt(var_proposal)
            reject =  self.is_rejected(proposal)
            iter += 1

        if reject:
            self.fail += 1
        else:
            self.stochastic.value = proposal
#        kabuki.debug_here()

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

class UninformativePriorNormalstd(PriorNormalstd):
    """
    Step method for Uniform Prior over standard devision of Normal distribution
    using reject sampling
    """

    target_class = pm.Uniform

    def __init__(self, stochastic, **kwargs):
        PriorNormalstd.__init__(self, stochastic, **kwargs)

    def is_rejected(self, proposal):
        return False


class MetropolisAlpha(pm.Metropolis):
    """
    step method for the alpha in SPX
    """

    def __init__(self, stochastic, betas, loc, scale, *args, **kwargs):
        pm.Metropolis.__init__(self, stochastic, *args, **kwargs)
        self.betas = betas
        self.scale = scale
        self.loc = loc
        self.alpha = stochastic

        #set self.children
        for beta in self.betas:
            self.children.update(beta.extended_children)


    def _get_logp_plus_loglike(self):
        return pm.logp_of_set(self.children)

    logp_plus_loglike = property(fget = _get_logp_plus_loglike)


    def step(self):
        """
        The default step method applies if the variable is floating-point
        valued, and is not being proposed from its prior.
        """

        # Probability and likelihood for s's current value:
        logp = self.logp_plus_loglike

        # Sample a candidate value
        self.propose()

        # Probability and likelihood for s's proposed value:
        try:
            logp_p = self.logp_plus_loglike

        except ZeroProbability:

            # Reject proposal
            self.reject()

            # Increment rejected count
            self.rejected += 1

            return

        # Evaluate acceptance ratio
        if np.log(np.random.rand()) > logp_p - logp:

            # Revert s if fail
            self.reject()

            # Increment rejected count
            self.rejected += 1
        else:
            #update all the other variables
            for node in self.loc:
                node.value  = node.value * self.alpha_ratio
            self.scale.value = self.scale.value * np.abs(self.alpha_ratio)

            # Increment accepted count
            self.accepted += 1


    def propose(self):
        """
        This method is called by step() to generate proposed values
        if self.proposal_distribution is "Normal" (i.e. no proposal specified).
        """

        last_value  = self.alpha.value
        self.alpha.value = np.random.normal(self.alpha.value, self.adaptive_scale_factor * self.proposal_sd,
                                        size=self.alpha.value.shape)
        self.alpha_ratio = self.alpha.value / last_value
        for beta in self.betas:
            beta.value = beta.value * self.alpha_ratio

    def reject(self):
        # Sets current s value to the last accepted value
        # self.stochastic.value = self.stochastic.last_value
        self.stochastic.revert()
        for beta in self.betas:
            beta.revert()



class SPXcentered(pm.StepMethod):
    """
    PX step method for centered data:
    y_ij ~ f(subj_j)
    subj_j ~ g(loc, scale)
    g has to have the property that if X ~ g(a,b) than k*X ~ g(k*a,k*b)
    """

    def __init__(self, beta, loc, scale, loc_step_method=pm.Metropolis,
                 scale_step_method=pm.Metropolis, beta_step_method=pm.Metropolis,
                *args, **kwargs):

        if type(loc) != list:
            loc = [loc]
        pm.StepMethod.__init__(self, [scale] + loc + beta, *args, **kwargs)

        #set nodes
        self.alpha = pm.Uninformative('alpha', value=1., trace=False, plot=False)
        self.loc = loc
        self.scale = scale
        self.beta = set([])
        for node in self.loc:
            self.beta.update(node.extended_children)

        #set step methods
        self.loc_steps = [loc_step_method(node) for node in self.loc]
        self.scale_step = scale_step_method(scale)

        self.beta_steps = [beta_step_method(node) for node in self.beta]
        self.alpha_step = MetropolisAlpha(self.alpha, self.beta, loc, scale)


    def step(self):

        #take one step for all the nodes
        [x.step() for x in self.beta_steps]
        [x.step() for x in self.loc_steps]
        self.scale_step.step()

        #take a step for alpha
        self.alpha_step.step()

    def tune(self, verbose):
        #tune scale
        tuning = self.scale_step.tune()

        #tune beta
        for step in self.beta_steps:
            tuning = tuning | step.tune()

        #tune loc
        for step in self.loc_steps:
            tuning = tuning | step.tune()

        #tune alpha
        tuning = tuning | self.alpha_step.tune()

        return tuning
