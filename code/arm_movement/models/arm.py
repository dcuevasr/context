# -*- coding: utf-8 -*-
# ./arm.py

"""Functions regarding the simulation of the arm as an attractor."""

import itertools as it
import ipdb

import numpy as np
from scipy.stats import multivariate_normal
from scipy.integrate import solve_bvp, solve_ivp
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt


class PriorArm(object):
    """Prior-based control for an N-Dimensional arm."""

    def __init__(self, x_ini=None, x_end=None, prior_fun=None):
        """Well..."""
        self.ndim = 2  # Dimensions of the movement (2D/3D)
        self.deg_free = 2  # Degrees of freedom
        self.default_smoothness = 1  # Default value of the second deriv. of
        # position
        self.default_velocity = None  # Default velocity. "None" starts all
        # movements in the direction of the goal.
        self.flag_changed_priors = False
        self._comfort = self._create_priors()
        self.priors_fun = self._comfort
        if x_ini is None:
            self.x_ini = np.zeros(self.ndim)
            self.x_ini[0] = 0.5
        else:
            self.x_ini = x_ini
        if x_end is None:
            self.x_end = np.zeros(self.ndim)
            self.x_end[1] = 0.5
        else:
            self.x_end = x_end

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

        return prior_diff_fun

    def visualize_priors(self, num_points=100, fignum=1):
        """If self.ndim == 2, plots a heat map of the priors for visualization.
        Note that it can only take the priors functions from self.

        """
        if self.ndim != 2:
            raise ValueError('The space needs to be 2D for this function.')

        mesh = np.zeros([num_points, num_points])

        for idx, idy in it.product(range(num_points), range(num_points)):
            mesh[idx, idy] = self.priors_fun(
                [idx / num_points, idy / num_points])

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
            mesh[trial, :] = self.priors_fun(coords[trial, :])
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

    def _ode(self, time, x, m=(3,)):
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
        ndim = self.ndim
        if x.ndim == 1:
            x = x[:, None]
        from_priors = np.array([self.priors_fun(x[:ndim, idx])
                                for idx in range(x.shape[-1])])
        out = (x[ndim:, :], (m[0] * (self.x_end - x[:ndim, :].T) + from_priors).T)
        return np.squeeze(np.vstack((out)))

    def solve_boundary_value(self,):
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
        \]
        Note that y_1, y_2 \in R^2, which makes this a 4d system, although the
        last two dimensions are of no use to us.
        """
        time_mesh = np.linspace(0, 1, 5)
        y_mesh = 0.5 * np.ones((2 * self.ndim, time_mesh.size))
        p_mesh = (1,)  # np.zeros(time_mesh.size)
        integrated = solve_bvp(self._ode, self._residuals,
                               time_mesh, y_mesh,  # p_mesh,
                               tol=0.001, bc_tol=0.0001)
        return integrated

    def _residuals(self, x_ini, x_end, m=None):
        """Residuals for the boundary-conditions-problem."""
        whys = np.hstack((np.abs(self.x_ini - x_ini[:2]) + np.abs(self.x_end - x_end[:2]),
                          np.abs(x_end[2:])))
        if m is not None:
            whys = np.hstack(whys, 1 - m)
        return whys

    def solve_initial_value(self, initial_speed=None, t_interval=None):
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
        if t_interval is None:
            t_interval = (0, 1)
        if initial_speed is None:
            initial_speed = self.x_end - self.x_ini
            initial_speed /= np.linalg.norm(initial_speed)
        x0 = np.concatenate((self.x_ini, initial_speed))
        integrated = solve_ivp(self._ode, t_interval, x0)  # rtol=1e90, atol=1e90, max_step=0.01)
        return integrated


class FirstOrderArm(PriorArm):
    """Version of the PriorArm with a 1st degree ODE."""
    def _ode(self, time, x, m=(3, (1, 1), 0.2)):
        r"""Defines the 1st order ODE:
        \[
        \dot x = m (g - x) + f(x) + h(t)
        \]
        where f(x) are the priors (self.prior_fun), $g$ the goal and $h(t)$
        a boost in the direction specified by --m--[1].

        Parameters
        ----------
        m : 3-tuple
        The first element is the strenght of the control. The second element
        should be a 1darray (or iterable) which is the initial speed of the
        problem to solve. This will be applied as a boost that lasts m[-1]
        seconds after the start of the solution. The third element (duh) is
        the duration of the boost.

        """
        ndim = self.ndim
        if x.ndim == 1:
            x = x[:, None]
        if time < m[-1]:
            pulse = np.array(m[1])
        else:
            pulse = 0

        from_priors = np.array([self.priors_fun(x[:, idx])
                                for idx in range(x.shape[-1])])
        out = m[0] * (self.x_end - x.T) + from_priors + pulse
        return out

    def solve_initial_value(self, initial_speed=None, t_interval=None):
        """Solves the _ode with the initial conditions."""
        if t_interval is None:
            t_interval = np.array([0, 1])
        if initial_speed is None:
            initial_speed = np.array([0.2, 0.2])
        x0 = self.x_ini
        integrated = solve_ivp(self._ode, t_interval,
                               args=((3, initial_speed, 0.2),),
                               y0=x0, dense_output=True)
        return integrated


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


def model_best_path(arm=None, prior_mu=None):
    """Sets up a pymc3 model for finding the best initial speed

    Parameters
    ----------
    arm : PriorArm instance or children
    Arm to use for simulations. The dimensions of the space (arm.ndim)
    must match the number of elements of --x_ini/end--. If no arm is
    provided, the default one is created (for testing).

    prior_mu : 1darray
    Vector with as many elements as spatial dimensions (arm.ndim) that
    determines the mean of the bounded Gaussian used for priors over
    the initial angle of movement. If None, a vector pointing from
    arm.x_ini to arm.x_end is used.

    Returns
    -------
    model : pm.Model instance
    Model with the required variables. Ready to do sampling. Low on
    sugar and vegan.
    """
    if arm is None:
        arm = PriorArm()
    x_end = arm.x_end
    x_ini = arm.x_ini
    if prior_mu is None:
        prior_mu = x_end - x_ini
    prior_angle_mean = np.arccos(prior_mu[0] / np.linalg.norm(prior_mu))
    prior_angle_sd = 0.5
    circle_normal = pm.Bound(pm.Normal, lower=0, upper=2 * np.pi)

    with pm.Model() as mamo:
        angle_ini = circle_normal('angle_ini', mu=prior_angle_mean,
                                  sd=prior_angle_sd,
                                  testval=np.pi / 4)
        x_ini = tt.stack([tt.cos(angle_ini), tt.sin(angle_ini)])
        pm.DensityDist('logp', arm_logli(arm), observed=arm.x_ini)
    return mamo


def _log_likelihood(arm, initial_speed):
    r"""Likelihood function to be used for inference. Solves the ODE
    with the initial conditions and checks whether the solution went
    near the goal. Returns a number that is higher the closer the
    solution gets to the goal, and the faster it does so, following
    the formula:
    \[
    L &= t_g - |g - x(t_g)| \\
    t_g &= min_{t \in (0, 1]}d(g, x(t))
    \]
    where $d()$ is the Euclidean distance.
    \]

    """
    epsi = 0.1
    times = np.linspace(0, 1, 30)
    # initial_speed = x_ini[arm.ndim:]
    integrated = arm.solve_initial_value(initial_speed=initial_speed)
    x_t = integrated.sol(times).T  # integrated.y[:arm.ndim, :].T
    ydiff = np.diff(x_t, axis=0)
    jerk = np.sum(np.abs(np.diff(ydiff[:, 0] / ydiff[:, 1])))
    distances = np.linalg.norm(arm.x_end - x_t, axis=1)
    ix_min = np.array(np.where(distances <= epsi)[0])
    if ix_min.size == 0:
        return -np.inf
    ix_min = ix_min[0]
    t_g = times[ix_min]  # integrated.t[ix_min]
    likelihood = - 0.1 * t_g - distances[ix_min] - 100 * jerk
    # ipdb.set_trace()
    return likelihood


class arm_logli(tt.Op):
    """Theano Op for the log-likelihood for inference in model_best_path()."""
    __props__ = ()
    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, arm):
        self.arm = arm

    def perform(self, node, inputs, output_storage):
        epsi = 0.1
        arm = self.arm
        initial_speed = inputs[0]
        # integrated = arm.solve_initial_value(initial_speed=initial_speed)
        # x_t = integrated.sol(np.linspace(0, 10, 20)).T  # integrated.y[:arm.ndim, :].T
        # distances = np.linalg.norm(arm.x_end - x_t, axis=1)
        # ix_min = np.array(np.where(distances <= epsi)[0])
        # if ix_min.size == 0:
        #     output_storage[0][0] = -np.inf
        #     return
        # ix_min = ix_min[0]
        #     # ix_min = np.argmin(distances)
        # t_g = integrated.t[ix_min]
        # likelihood = -10 * t_g - distances[ix_min]
        likelihood = _log_likelihood(arm, initial_speed)
        output_storage[0][0] = likelihood


def infer_best_path(x_ini=None, x_end=None, arm=None, **model_kwargs):
    """Infers the best path from --x_ini-- to --x_end-- using
    Metropolis sampling.

    Parameters
    ----------
    x_ini : 1darray
    Initial position. If None, its value is taken from the arm.

    x_end : 1darray
    Goal position. If None, its value is taken from the arm.

    arm : PriorArm instance
    Arm to use for the simulations. If None, the arm is created
    using PriorArm. If x_end or x_ini are provided, they are used
    to create the arm.

    **model_kwargs are those sent to model_best_path() (other than
    --arm--)

    """
    if arm is None:
        return_arm = True
        arm = PriorArm(x_ini=x_ini, x_end=x_end)
    else:
        return_arm = False
        if x_ini is None:
            x_ini = arm.x_ini
        else:
            arm.x_ini = x_ini
        if x_end is None:
            x_end = arm.x_end
        else:
            arm.x_end = x_end

    model = model_best_path(arm)
    with model:
        samples = pm.sample(200, step=pm.Metropolis())
    if return_arm:
        return samples, arm
    return samples


def plot_infer_best_path(samples, arm, fignum=3):
    """Plots the results from infer_best_path(). It takes as inputs
    the outputs of infer_best_path().
    """
    mean_angle = samples.get_values('angle_ini').mean()
    best_x_ini = np.array([np.cos(mean_angle), np.sin(mean_angle)])
    integrated = arm.solve_initial_value(initial_speed=best_x_ini)

    fig, axes = plt.subplots(1, 2, num=fignum, clear=True)

    axes[0].plot(*integrated.y[:2], marker='o')
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    axes[1].hist(samples.get_values('angle_ini'))
    plt.show(block=False)


def simulate_well_arm():
    """Runs simulations for motion planning for an arm in which there
    is an obstacle in the middle in the form of an unpassable ring.

    """
    arm = FirstOrderArm()
    well_fun_1 = _well_function(radius=0.1,
                                center=np.array((0.5, 0.3)),
                                slope=-15 / 0.05, alpha=25)

    well_fun_2 = _well_function(radius=0.1,
                                center=np.array([0.6, 0.7]),
                                slope=-15 / 0.05, alpha=25)

    def funny(x, le_fun=arm.priors_fun, well_fun_1=well_fun_1,
              well_fun_2=well_fun_2):
        return le_fun(x) + well_fun_1(x) + well_fun_2(x)
    arm.priors_fun = funny

    arm.visualize_prior_vector_field(num_points=50)

    samples = infer_best_path(arm=arm, x_end=np.array((0.2, 0.3)),
                              prior_mu=None)#np.array([-0.1, 0.1]))

    plot_infer_best_path(samples=samples, arm=arm)
    return arm, samples


def _well_function(radius=0.02, slope=-30, alpha=5, center=(0, 0)):
    r"""Returns a function for the well used in simulate_well_arm().

    The well is centered at --center--, and is angle-invariant. Its
    height (that is, the strength with which it repells) depends on
    the distance from the center:
    \[
    f(r) = \alpha               for  r < r_0
           m(r - r_0) + \alpha  for  r >= r_0
           0                    for  r >= (r_0 - \alpha) / m
    \]
    where $r = ||x - x_0||$, r_0 is --radius--, m is --slope--, and
    x_0 is --center--.

    Returns
    -------
    well : function
    Function with signature well(x), where x are the Cartesian coordinates
    at which the function whould be evaluated.

    """
    center = np.array(center)

    def well(x):
        r = np.linalg.norm(x - center)
        if r == 0:
            return np.zeros_like(x)
        theta = np.arccos((x[0] - center[0]) / r)
        if r < radius:
            out = alpha
        else:
            out = max(0, slope * (r - radius) + alpha)
        ret_val = out * (x - center) / r
        return ret_val
    return well


def simple_integration():
    """Integrates the PriorArm._ode() function from 0 to 1 using the dumbest
    x += dt * f(x) integration method to compare against scipy's solve_ivp.

    """
    arm = PriorArm(x_end=[0.5, 1])
    arm.priors_fun = lambda x: 0
    initial_speed = np.array([0, 0.2])

    T = 5
    dt = 0.001
    x = np.array([*arm.x_ini, *initial_speed])
    x_t = np.zeros((x.size, np.ceil(T / dt).astype(int) + 1))
    for idt, t in enumerate(np.arange(0, T, dt)):
        x += dt * arm._ode(t, x, m=(5,))
        x_t[:, idt] = x
    return arm, x_t
