# -*- coding: utf-8 -*-
# ./adaptation/model.py

"""Bayesian motor adaptation model."""


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
