# -*- coding: utf-8 -*-
# ./arm.py

"""Functions regarding the simulation of the arm as an attractor."""

import numpy as np
import matplotlib.pyplot as plt

import attractors as att


class PriorArm(object):
    """Prior-based control for an N-Dimensional arm."""
    def __init__(self,):
        """Well..."""
        self.ndim = 2  # Dimensions of the movement (2D/3D)
        self.deg_free = 2  # Degrees of freedom
        self.default_smoothness = 1  # Default value of the second deriv. of
                                     # position
        self.default_velocity = None  # Default velocity. "None" starts all
                                      # movements in the direction of the goal.
        self.flag_changed_priors = False
        self._comfort = self._create_priors()
        self.priors = self._comfort

    def priors_control(self, delta_x, speed):
        """Given a direction (i.e. an control state), defines a probability distribution
        over all possible ending points, given the current one.

        In this implementation, the probability distribution is divided into two
        iid components: the one in the direction of delta_x, and the one in the
        ortogonal subspace. In the delta_x direction, the distribution is a
        Gamma distribution, with shape and scale parameters determined by the
        velocity of the movement (given by --speed--), calculated in the method
        _dist_from_speed. In the ortogonal subspace, the distribution is a
        Gaussian with zero mean and a standard deviation that depends on
        --speed--, calculated in the method _dist_from_speed. One such Gaussian
        distribution is calculated for each dimension.

        """
        pass

    def _dist_from_speed(self, speed):
        """Returns the parameters of the distribution over next states given an
        action. This function assumes that the space is in local coordinates,
        with the first coordinate being that of the direction of movement, and
        the rest the ortogonal subspace.

        See the documentations of self.priors_control for more details on the
        distributions.

        For the gamma distribution, the parameters are calculated as follows:
        TODO

        For the ortogonal subspace, the standard deviations are all equal to the
        speed.

        """
        pass

    def _curr_distance(self, curr_pos, goal):
        """Calculates the Euclidean distance between the current position (--curr_pos--)
        and the goal (--goal--).

        Returns
        -------
        distance : float
        Euclidean distance.

        """
        return distance

    def _create_priors(self, filename="priors_control.pi"):
        """Either loads the priors from --filename-- or, if the file does not exist or a
        filename is not provided (None), creates them from scratch and saves
        them to --filename--.

        Note that these priors must be in the Euclidean coordinates of the
        position of the hand.

        In this implementation, the priors are a sum of multivariate Gaussians
        with fancy-ass covariance matrices.

        """
        self.prior_pars = [["mu", "sigma"], [[], []]]
        self.priors = priors
        pass

    def _diff_priors(self, ):
        """Differential of the priors. Relies on self.prior_pars, which is created
        in _create_priors.

        Saves
        -----
        diff_priors : function
        Function of the position x. Returns a normalized vector in the direction of
        the gradient of the priors.

        """
        pass

    def add_obstacles(self, obstacles):
        """Mixes the probability distribution of the priors (self.priors) with that
        given in --obstacles--. If this has already been done before, it will do
        nothing. The priors can be reset to their original value by running
        self.reset_priors first.

        """


        self.priors = new_priors
        self.flag_changed_priors = True
        pass

    def reset_priors(self,):
        """Resets priors that have been changed via add_obstacles().

        """
        self.priors = self._comfort
        self.flag_changed_priors = False
