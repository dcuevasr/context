# -*- coding: utf-8 -*-
# code/shs.py


"""Defines classes for stable heteroclinic sequences."""

import numpy as np
from scipy import integrate
import pandas as pd


class Attractor(object):
    """Base class for attractor dynamics. Not meant to be used directly

    Subclasses should define the following methods:

    x_dot(t, y) : rhs of the ODE, where --t-- is time and --y-- is
    n-dimensional.

    """

    def __init__(self, *odeargs, **odekwargs):
        """Cries in silence."""
        pass

    def x_dot(self, t, x):
        r"""This is the expected call signature of x_dot. Define it in
        subclasses or face the dire consequences.

        Should return the rhs of \dot x = f(x,t).

        """
        pass

    def set_initial_conditions(self, ):
        """Sets random initial conditions in the interval [0, 1]. """
        return np.random.rand(self.N)

    def integrate(self, t_end, x_ini=None, t_ini=0):
        """Integrates the ODE using the initial condition --x_init--, from t=0 to
        t = t_end.

        Parameters
        ----------
        x_init : ndarray
        Initial condition. Must have size of self.N. If not provided,
        set_initial_conditions() is called.

        t_end/ini float
        Final/initial time for integration. t_init defaults to whatever it was
        before, which could be the last point of integration if this function
        has been called before.

        Returns
        -------
        trajectory : 3darray
        Pandas DataFrame with columns 't' (for time points) and 'xA', where
        A is in [0, N), for each of the dimensions of the space.

        """
        if x_ini is None:
            x_ini = self.set_initial_conditions()
        output = integrate.solve_ivp(self.x_dot, [t_ini, t_end], x_ini)
        inte_t = output['t']
        inte_x = output['y']
        labels = ['x{}'.format(ix_x) for ix_x in range(self.N)]
        pandata = pd.DataFrame(inte_x.T, columns=labels)
        pandata['t'] = inte_t
        return pandata

    def next_point(self, curr_x, delta_t, t_ini=None):
        r"""Returns a simulated next point, using:
        x(t+dt) = x(t) + \dot x(t) * dt.

        """
        gradient = self.x_dot(t=t_ini, x=curr_x)
        return curr_x + gradient * delta_t


class PointAttractor(Attractor):
    r"""Point attractor dynamics of the form:
    \[
    \dot x = \eta x
    \]
    where $x \in R^n$, and $\eta \in R^n$ are parameters to be set in
    initialization. Note that for $\eta < 0$, the attractor is repulsive, like
    yo mamma.

    """
    def __init__(self, pos=(0, 0), eta=(-1, -1)):
        """Sets up the differential equations, given the parameters.

        Parameters
        ----------
        pos : iterable
        Position (x1, x2, ..., xn) of the equilibrium point.

        eta : iterable
        Parameters of the equations. Its size must match that of --pos--.

        """
        self.pos = np.array(pos)
        self.eta = np.array(eta)
        if self.pos.shape != self.eta.shape:
            raise ValueError('The sizes of --pos-- and --eta-- do not match.')
        self.N = len(pos)  # Size of the space

        # Good default \delta t for simulations
        self.default_dt = 0.01

    def x_dot(self, t=0, x=0):
        """Returns f(curr_x), where f is the rhs of the ODE.

        Parameters
        ----------
        curr_x : ndarray
        Initial condition of x. Must have the right size (self.N).

        t : float
        Does nothing. Included only for scipy's ODE integrator compatibility

        """
        return self.eta * (x - self.pos)

    def set_pos(self, new_pos):
        """Changes the position of the equilibrium point."""
        if self.pos.shape != new_pos.shape:
            raise ValueError('New equilibrium point is of a different size '
                             'as the old one.')
        self.pos = new_pos


class Shs(Attractor):
    r"""Attractor dynamics with stable heteroclinic sequences based on the
    Lotka-Volterra equations, using the parametrization from AZR* which
    guarantees the existence and stability of the SHS.

    The system follows the following ODE:
    \[
    \dot x_i = x_i\left(\sigma _i + \rho _{ij}x_j\right)
    \]
    using Einstein's notation for summation.


    * Afraimovich, V S, V P Zhigulin, and M I Rabinovich. “On the Origin of
      Reproducible Sequential Activity in Neural Circuits.” Chaos (Woodbury,
      N.Y.) 14, no. 4 (December 2004):
      1123–29. https://doi.org/10.1063/1.1819625.

    """
    def __init__(self, num_neurons, sequence):
        """Initializes the parameters of the LV equations.

        Parameters
        ----------
        num_neurons : int
        Size of the space (e.g. number of dimensions).

        sequence : 1darray size <= num_neurons
        Sequence to hard-code into the system, represented as a sequence of
        ints. The system will have len(sequence) equilibrium points of the
        form (0, 0, ..., 1, ..., 0), where the 1 is in the i-th position, one
        for each member of --sequence--.

        """
        self.N = num_neurons
        self.sequence = sequence
        self._set_sigmas()
        self._set_rhos()

    def _set_sigmas(self,):
        """Sets random values for sigma, ensuring that the sigma[0] >= sigma[j]
        for all j.

        Sets
        ----
        sigma : 1darray size == [N,]
        Random values of the sigma parameters.

        """
        sigma = 10 + 5 * np.random.rand(self.N)
        old_zero = sigma[0]
        ix_max = sigma.argmax()
        sigma[0] = sigma[ix_max]
        sigma[ix_max] = old_zero
        self.sigma = sigma

    def _set_rhos(self, ret_rho=False, set_rho=True):
        """Sets the connectivity matrix for the N neurons.

        Parameters
        ----------
        sigma : 1darray size==[N, ]
        ODE parameters

        sequence : 1darray size <= N
        Desired sequence to be written into the ODE. Each neuron can be
        included at most once.

        Sets
        ----
        rho : 2darray size == [N, N]
        Connection strenght between all neurons.

        """
        sigma = self.sigma
        sequence = self.sequence
        rho = 2 * np.ones([self.N, self.N], dtype=np.float)

        # Condition 41 AZR
        for p_seq, c_seq in zip(sequence, sequence[1:]):
            rho[p_seq, c_seq] = sigma[p_seq] / sigma[c_seq] + 0.51

        np.fill_diagonal(rho, 1)

        # Condition 42 AZR
        for c_seq, n_seq in zip(sequence, sequence[1:]):
            rho[n_seq, c_seq] = sigma[n_seq] / sigma[c_seq] - 0.5

        for ixx in range(self.N):
            for ixy in range(1, len(sequence)):
                cond_1 = ixx != sequence[ixy - 1]
                cond_2 = ixy == len(sequence) - 1
                if cond_2:
                    cond_3 = True  # Dummy value
                else:
                    cond_3 = ixx != sequence[ixy + 1]
                if cond_1 and (cond_2 or cond_3):
                    sum_1 = rho[sequence[ixy - 1], sequence[ixy]]
                    sum_2 = (sigma[ixx] - sigma[sequence[ixy - 1]])
                    div = sigma[sequence[ixy]]
                    rho[ixx, sequence[ixy]] = sum_1 + sum_2 / div + 2
        np.fill_diagonal(rho, 1)
        if set_rho:
            self.rho = rho
        if ret_rho:
            return rho

    def set_initial_conditions(self, background_noise=0.01):
        """Returns a set of good initial conditions (i.e. around the first equilibrium
        point, and already going towards the second one).

        """
        x_init = background_noise * np.ones(self.N)
        x_init[self.sequence[0]] = 0.8 * self.sigma[self.sequence[0]]
        x_init[self.sequence[1]] = 0.2 * self.sigma[self.sequence[1]]
        return x_init

    def x_dot(self, t, x):
        """Rhs of the Lotka-Volterra ODE.

        Parameters
        ----------
        t : float
        Does nothing. Included only for scipy's ODE integrator compatibility
        """
        return x * (self.sigma - self.rho.dot(x))
