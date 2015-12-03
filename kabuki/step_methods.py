import pymc as pm
import numpy as np
import kabuki
from numpy.random import randn, rand
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
        assert isinstance(self.stochastic, pm.Normal)

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
    Step method for Uniform Prior over standard deviation of Normal distribution
    using reject sampling
    f(sigma) = f(var)*|2*sigma| = InverseGamma(alpha, beta) = 1./ Gamma(alpha, beta)
    alpha=(n-1)/2,   beta=sum(x_i-mu)**2/2
    so we sample the an r.v. from Gamma and take its inverse, which gives us var,
    and then we take the square root of it.
    """

    child_class = pm.Normal
    parent_label = 'tau'

    def __init__(self, stochastic, maxiter=100, **kwargs):
        pm.Gibbs.__init__(self, stochastic, **kwargs)
        self.maxiter = maxiter
        n = sum([len(x.value.flatten()) for x in self.children])
        self.alpha = (n - 1) / 2.
        self.mu_nodes = np.unique([x.parents['mu'] for x in self.children])
        self.fail = 0

        self.groups = [None]*len(self.mu_nodes)
        for i, mu in enumerate(self.mu_nodes):
            self.groups[i] = [x for x in self.children if x.parents['mu'] == mu]

    def step(self):

        #compute beta
        self.beta = 0
        for i, mu in enumerate(self.mu_nodes):
            if isinstance(mu, pm.Node):
                mu_val = mu.value
            else:
                mu_val = mu
            self.beta  += sum([sum((x.value.flatten() - mu_val)**2) for x in self.groups[i]])
        self.beta /= 2.

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

        except pm.ZeroProbability:

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

    def __init__(self, loc, scale, loc_step_method=None,
                 scale_step_method=None, beta_step_method=None,
                 loc_step_method_args=None, scale_step_method_args=None,
                 beta_step_method_args=None, *args, **kwargs):

        if type(loc) != list:
            loc = [loc]
        self.loc = loc
        self.scale = scale
        self.beta = set([])
        for node in self.loc:
            self.beta.update(node.extended_children)

        pm.StepMethod.__init__(self, [scale] + loc + list(self.beta), *args, **kwargs)

        #set alpha
        self.alpha = pm.Uninformative('alpha', value=1., trace=False, plot=False)

        #assign default Metropolis step method if needed
        if loc_step_method is None:
            loc_step_method = pm.Metropolis
        if scale_step_method is None:
            scale_step_method = pm.Metropolis
        if beta_step_method is None:
            beta_step_method = pm.Metropolis

        if loc_step_method_args is None:
            loc_step_method_args = {}
        if scale_step_method_args is None:
            scale_step_method_args = {}
        if beta_step_method_args is None:
            beta_step_method_args = {}

        #set step methods
        self.loc_steps = [loc_step_method(node, **loc_step_method_args) for node in self.loc]
        self.scale_step = scale_step_method(scale, **scale_step_method_args)
        self.beta_steps = [beta_step_method(node, **beta_step_method_args) for node in self.beta]
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


class SliceStep(pm.Gibbs):
    """
    simple slice sampler
    """
    def __init__(self, stochastic, width = 2, maxiter = 200, left = None,
                 verbose = -1, *args, **kwargs):
        """
        Input:
            stochastic - stochastic node
            width <float> - the initial width of the interval
            maxiter <int> - maximum number of iteration allowed for stepping-out and shrinking
            left <int> - the starting position of the interval (default is None).
        """
        pm.Gibbs.__init__(self, stochastic, verbose=verbose, *args, **kwargs)
        self.width = width
        self.neval = 0
        self.maxiter = maxiter
        self.left = left

    def step(self):
        stoch = self.stochastic
        value = stoch.value

        #sample vertical level
        z = self.logp_plus_loglike - np.random.exponential()

        if self.verbose>2:
            print(self._id + ' current value: %.3f' % value)
            print(self._id + ' sampled vertical level ' + repr(z))


        #position an interval at random starting position around the current value
        r = self.width * np.random.rand()
        xr = value + r
        if self.left is not None:
            xl = self.left
        else:
            xl = xr - self.width


        if self.verbose>2:
            print('initial interval [%.3f, %.3f]' % (xl, xr))

        if self.left is None:
            #step out to the left
            iter = 0
            stoch.value = xl
            while (self.get_logp() >= z) and (iter < self.maxiter):
                xl -= self.width
                stoch.value = xl
                iter += 1

            assert iter < self.maxiter, "Step-out procedure failed"
            self.neval += iter

            if self.verbose>2:
                print('after %d iteration interval is [%.3f, %.3f]' % (iter, xl, xr))

        #step out to the right
        iter = 0
        stoch.value = xr
        while (self.get_logp() >= z) and (iter < self.maxiter):
                xr += self.width
                stoch.value = xr
                iter += 1

        assert iter < self.maxiter, "Step-out procedure failed"
        self.neval += iter
        if self.verbose>2:
            print('after %d iteration interval is [%.3f, %.3f]' % (iter, xl, xr))

        #draw a new point from the interval [xl, xr].
        xp = rand()*(xr-xl) + xl
        stoch.value = xp

        #if the point is outside the interval than shrink it and draw again
        iter = 0
        while(self.get_logp() < z) and (iter < self.maxiter):
            if (xp > value):
                xr = xp
            else:
                xl = xp
            xp = rand() * (xr-xl) + xl #draw again
            stoch.value = xp
            iter += 1

        assert iter < self.maxiter, "Shrink-in procedure failed."
        self.neval += iter
        if self.verbose>2:
            print('after %d iteration found new value: %.3f' % (iter, xp))


    def get_logp(self):
        try:
            return self.logp_plus_loglike
        except pm.ZeroProbability:
            return -np.inf
