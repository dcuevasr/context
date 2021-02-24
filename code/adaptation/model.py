# -*- coding: utf-8 -*-
# ./adaptation/model.py

"""Bayesian motor adaptation model."""


class RandomVariable(Object):
    """Defines a random variable, with a pdf and cdf, as well as a __call__ method
    that samples from the pdf.

    The distribution parameters can be updated via the method update_parameters.

    Can be initiated with its own pdf and cdf. If None provided, it defaults to
    numpy's implementation of a Gaussian.

    """

    def __init__(self,):
        """Does nothing!"""
        pass

    def __call__(self, ):
        return self.sample()

    def sample(self, ):
        """Samples a value using the pdf."""
        rand = np.random.rand()
        if self.var_type == 'continuous':
            solution = sp.optimize.minimize(fun=lambda x: self.cdf(x) - rand,
                                            x0=0)
            sample = solution.x
        else:
            sample = np.argmin(abs(self.cdf - rand))
        return sample

    def plot_pdf(self, num_points=100, support=None):
        """Plots a smooth approximation of the pdf, with --num_points-- points and
        an x-range given by the elements of --support--.
        """
        raise NotImplementedError('Not yet!')


class GaussGaussInference(RandomVariable):
    """Performs Bayesian inference using Gaussian priors and likelihood. """
    def __init__(self, mu=None, sd=None):
        """Initiates the Gaussian priors with the given parameters or defaults.
        """
        if mu is None:
            mu = 0
        if sd is None:
            sd = 1

        self.pars = np.array([mu, sd])
        self.pars_history = [self.pars]

    def pdf(self, x):
        """Evaluates the Gaussian pdf on the current parameters."""
        return sp.stats.norm.pdf(x, loc=self.pars[0], scale=self.pars[1])

    def cdf(self, x):
        """Evaluates the Gaussian cdf on the current parameters."""
        return sp.stats.norm.cdf(x, loc=self.pars[0], scale=self.pars[1])

    def inference(self, likelihood_pars):
        """Updates the parameters of the posterior."""
        if priors_pars is None:
            priors_pars = self.pars
        mu_l, sd_l = likelihood_pars
        mu_p, sd_p = self.pars
        mu_post = (mu_l * sd_p ** 2 + mu_p * sd_p ** 2) / \
            (sd_l ** 2 + sd_p ** 2)
        sd_post = np.sqrt((sd_l ** 2 * sd_p ** 2) / (sd_p ** 2 + sd_l ** 2))
        self.pars = np.array([mu_post, sd_post])
        self.pars_history.append(self.pars)


class Context(Object):
    """Stores the force and direction of a context. If called, returns the force in
    2D Cartesian coordinates.

    """

    def __init__(self, magnitude=1, angle=0, baseline=False):
        """Well..."""
        self.magnitude = magnitude
        self.angle = angle

    def __call__(self, x=None):
        """Returns the force."""
        if baseline:
            return 0
        return self.magnitude * np.array([np.cos(self.angle),
                                          np.sin(self.angle)])

    def update(self, magnitude=None, angle=None):
        """Replaces the values of the magnitude and the angle, if
        any provided. If both are None, does nothing but cry and
        drink itself silly in the dark.

        """
        if magnitude:
            self.magnitude = magnitude
        if angle:
            self.angle = angle


class BayesianAgent(Object):
    """Base class of the agent that adapts to new motor dynamics."""

    def __init__(self, context_pars=None):
        self.num_contexts = 3
        if context_pars is None:
            context_pars = self._default_pars()
        contexts = self.init_contexts(context_pars)
        self.p_contexts = self._prior_context()  # Will be updated
        self.gamma = GaussGaussInference()

    def _default_pars(self, ):
        """Default parameters for the contexts, which are assumed to be like the
        baseline."""
        return [[0, 0]] * self.num_contexts

    def _prior_context(self, ):
        """Prior probabilities of observing all contexts. The baseline context
        is arbitrarily set to be five times as likely as any other.

        """
        return np.array([5, *np.ones(self.num_contexts - 1)])

    def _prior_force_magnitudes(self, ):
        """Defines the parameters for the normal distribution over the magnitude
        parameter of the force for each context. The priors are chosen to be
        centered around 2 with a standard deviation of 1.

        """
        return [[0, 0.01]] + [[2, 1]] * (self.num_contexts - 1)

    def init_contexts(self, context_pars=None):
        "Initializes a context with default values, or those in --context_pars--, if
        given.

        Returns a list of instances of the class Context, where the
        zero-th entry is the baseline.

        """
        contexts = [Context(baseline=True), ]
        for pars in context_pars:
            contexts.append(Context(pars))
        return contexts


class LeftRightAgent(Object):
    """Subclass that knows two contexts(plus baseline): one pointing left and one
    pointing right. The magnitude of these forces are inferred during the
    experiment.

    """
    def __init__(self,):
        """Initializes the known left and right contexts, as well as baseline."""
        pass

    def __init_contexts(self, ):
        """Creates baseline, left and right contexts, with known force direction
        but unknown magnitude.

        """
        
    def _prior_force_magnitudes(self, ):
        """Agent-specific contexts."""
        return [[0, 0.1], [2, 1], [2, 1]]
