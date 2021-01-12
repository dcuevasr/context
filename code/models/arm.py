# -*- coding: utf-8 -*-
# ./arm.py

"""Functions regarding the simulation of the arm as an attractor."""

import itertools as it

import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt


class PriorArm(object):
    """Prior-based control for an N-Dimensional arm."""
    def __init__(self, prior_fun=None):
        """Well..."""
        self.ndim = 2  # Dimensions of the movement (2D/3D)
        self.deg_free = 2  # Degrees of freedom
        self.default_smoothness = 1  # Default value of the second deriv. of
                                     # position
        self.default_velocity = None  # Default velocity. "None" starts all
                                      # movements in the direction of the goal.
        self.flag_changed_priors = False
        self._comfort = self._create_priors()
        self.priors_fun, self.priors_diff_fun = self._comfort
        self.x_ini = np.zeros(self.ndim)
        self.x_ini[0] = 0.5
        self.x_end = np.zeros(self.ndim)
        self.x_end[1] = 0.5

    def _curr_distance(self, curr_pos, goal):
        """Calculates the Euclidean distance between the current position (--curr_pos--)
        and the goal (--goal--).

        Returns
        -------
        distance : float
        Euclidean distance.

        """
        return np.linalg.norm(curr_pos - goal)

    def _create_priors(self, filename=None):
        """Either loads the priors from --filename-- or, if the file does not exist or a
        filename is not provided (None), creates them from scratch and saves
        them to --filename--.

        Note that these priors must be in the Euclidean coordinates of the
        position of the hand.

        If no filename is provided, the default priors are created by
        self._default_priors.

        """
        if filename:
            raise NotImplementedError('Oops, cannot load priors from file yet')
        return self._default_priors()

    def _default_priors(self,):
        """Creates the default priors when no filename for the priors was provided
        at instanciation time.

        The default priors are a sum of one Gaussian with mean at every corner
        of the hyper-cube (or square) of the space and a standard deviation of
        one.  This creates a rounded cross in the middle, where it's the most
        comfortable.

        Returns
        -------
        priors : function handle
        Function with a single input, which is the n-dimensional vector at which
        the priors should be evaluated.

        """
        prior_funs = []
        prior_diffs = []
        variance_matrix = np.array([[1, 7], [0.7, 1]])
        for mean in it.product(*[[0, 1]] * self.ndim):
            mean = np.array(mean)
            # variance_matrix = 0.01 * np.identity(self.ndim)
            inv_variance = np.linalg.inv(variance_matrix)
            temp_dist = multivariate_normal(mean, variance_matrix)
            temp_fun = temp_dist.pdf

            def temp_diff(x, mean=mean, temp_fun=temp_fun):
                """Note the negative value at the end. This is to make the
                corners repelling.

                Do not remove the mean=mean part. It prevents late binding
                """
                temp_coeff_diff = - np.dot(inv_variance, x - mean)
                out_value = temp_fun(x) * temp_coeff_diff
                return -out_value

            prior_funs.append(temp_fun)
            prior_diffs.append(temp_diff)

        def prior_fun(x):
            return np.sum([one_fun(x) for one_fun in prior_funs])

        def prior_diff_fun(x):
            return -np.sum([one_fun(x) for one_fun in prior_diffs], axis=0)

        return prior_fun, prior_diff_fun

    def visualize_priors(self, num_points=100, fignum=1):
        """If self.ndim == 2, plots a heat map of the priors for visualization.
        Note that it can only take the priors functions from self.

        """
        if self.ndim != 2:
            raise ValueError('The space needs to be 2D for this function.')

        mesh = np.zeros([num_points, num_points])

        for idx, idy in it.product(range(num_points), range(num_points)):
            mesh[idx, idy] = self.priors_fun([idx / num_points, idy / num_points])

        plt.figure(num=fignum, clear=True)
        plt.imshow(mesh, cmap='gray')
        plt.colorbar()
        plt.show(block=False)

    def visualize_prior_vector_field(self, num_points=20, fignum=2):
        """Visualizes the vector field created by the gradient of the priors.

        """
        if self.ndim != 2:
            raise ValueError(
                'How am I supposed to plot {} dimensions?'.format(self.ndim))
        mesh = np.zeros([num_points * num_points, 2])
        coords = np.zeros_like(mesh)
        for trial, (idx, idy) in enumerate(it.product(range(num_points), range(num_points))):
            coords[trial, :] = [idx / num_points, idy / num_points]
            mesh[trial, :] = self.priors_diff_fun(coords[trial, :])
        plt.figure(num=fignum, clear=True)
        plt.quiver(*coords.T, *mesh.T)
        plt.show(block=False)

    def add_obstacles(self, obstacles):
        """Mixes the probability distribution of the priors (self.priors) with that
        given in --obstacles--. If this has already been done before, it will do
        nothing. The priors can be reset to their original value by running
        self.reset_priors first.

        """
        self.flag_changed_priors = True
        pass

    def reset_priors(self,):
        """Resets priors that have been changed via add_obstacles().

        """
        self.priors = self._comfort
        self.flag_changed_priors = False

    def _ode(self, time, x, m=1):
        r"""Defines the differential equation to follow, of the form:
        \[
        \dot x = c(x, g) + f(x)
        \]
        where t is time, g is the goal (--goal--) and f(x) is the force exerted
        by the priors and obstacles. c is the control function, which in this
        implementation is simply $c(x, g) = g - x$.

        Parameters
        ----------
        x : 1darray
        Current position. Should be of size self.ndim, but this is not enforced.

        goal : 1darray
        Goal (final position). Must be the same size as --x-- or people die.
        They just drop dead. (Note: our engineers have suggested that people
        dropping dead may not in fact have anything to do with this, and instead
        it's just something people do sometimes)

        """

        from_priors = np.array([self.priors_diff_fun(x[0:2, idx])
                                for idx in range(x.shape[-1])])
        return np.vstack((x[2:4, :],
                          (m[0] * (self.x_end - x[0:2, :].T) + from_priors).T))

    def solve_boundary_value(self, x_ini, velocity_ini, goal):
        r"""Simulates a run given the initial parameters.

        The system to solve is the following:
        \[
        \ddot{x} - m(g - x) - p(x) = 0
        \]
        where $x$ is position, $m$ some parameter, $g$ the goal and p(x)
        the priors.

        However, to solve the boundary condition problem of starting at 0 and
        ending at g, the following change of variable is performed:
        \[
        y_1 &= \dot{y_2} \\
        y_2 &= m(g - y_1) + p(y_1)
        \]
        which becomes a system of 2 ODEs to solve under the boundary conditions:
        \[
        y_1(0) &= 0 \\
        y_1(T) &= g
        Note that y_1, y_2 \in R^2, which makes this a 4d system, although the
        last two dimensions are of no use to us.
        """
        time_mesh = np.linspace(0, 1, 5)
        y_mesh = 0.5 * np.ones((2 * self.ndim, time_mesh.size))
        p_mesh = (1,)  # np.zeros(time_mesh.size)
        integrated = solve_bvp(self._ode, self._residuals,
                               time_mesh, y_mesh, p_mesh,
                               tol=0.001, bc_tol=0.0001)
        return integrated

    def _residuals(self, x_ini, x_end, m=0):
        """Residuals for the boundary-conditions-problem."""
        return np.hstack(((self.x_ini - x_ini[:2]) + (self.x_end - x_end[:2]),
                          np.zeros(2), 0))

    def solve_initial_value(self, initial_speed=None):
        r"""Given initial conditions for x and x', solves the following system:
        \[
        \ddot{x} - m(g - x) - p(x) = 0
        \]
        where $x$ is position, $m$ some parameter, $g$ the goal and p(x)
        the priors.

        However, to solve the initial condition problem of starting at 0, the
        following change of variable is performed:
        \[
        y_1 &= \dot{y_2} \\
        y_2 &= m(g - y_1) + p(y_1)
        \]
        which becomes a system of 2 ODEs. The intial conditions are then:
        \[
        y_1(0) &= 0 \\
        y_2(0) &= s
        Where $s$ is the initial speed (input to this method).

        Note that y_1, y_2 \in R^2, which makes this a 4d system, although the
        last two dimensions are of no use to us.

        Parameters
        initial_speed : 1daray
        Vector with 2 elements representing the initial speed of the system.
        Defaults to a unit vector pointing towards the goal.
        """
        if initial_speed is None:
            initial_speed = self.x_end - self.x_ini
            initial_speed /= np.linalg.norm(initial_speed)
        x0 = np.concatenate((self.x_ini, initial_speed))
        x0p = None  # Unfinished


def visualize_multivariate(mean=(0, 0), cov=None, num_points=100, fignum=3):
    """Plot the pdf of a multivariate_normal. """
    if cov is None:
        cov = np.identity(2)

    pdf = multivariate_normal(mean, cov=cov).pdf
    mesh = np.zeros([num_points, num_points])

    for idx, idy in it.product(range(num_points), range(num_points)):
        point = 10 * (2 * np.array([idx, idy]) / num_points - 1)
        mesh[idx, idy] = pdf(point)

    plt.figure(num=fignum, clear=True)
    plt.imshow(mesh, cmap='gray')
    plt.draw()
    plt.colorbar()
    plt.show(block=False)
