# -*- coding: utf-8 -*-
# ../simulations.py


"""Collections of simulations "sin ton ni son"."""

import logging
import ipdb

from matplotlib import pyplot as plt
import matplotlib.animation as ani
import numpy as np
import pandas as pd

import attractors as att
import arm


# Do logging magic
handlers = logging.getLogger().handlers
stdout_handler = None
for handler in handlers:
    if isinstance(handler, logging.StreamHandler):
        stdout_handler = handler
        break
if stdout_handler is None:
    stdout_handler = logging.StreamHandler()
malogger = logging.getLogger(__name__)
malogger.handlers = [stdout_handler]
malogger.setLevel(logging.INFO)


def signal_next():
    """Simulates an SHS in which the trajectory to the next element in the sequence
    starts only after a signal has been sent to it. This is to simulate a
    higher-level monitoring system that determines when a subgoal (represented
    by an equilibrium point in the SHS) has been achieved, so the next one can
    start.

    To simulate a monitoring system, we assume that when the simulated
    trajectory is in a small neigtborhood of the target, there is a small chance
    that the monitor will determine it has accomplished the subgoal and send the
    signal to move on.

    A subgoal is said to be achieved if (goal - curr_x) / goal < epsilon, where
    goal and curr_x are current-subgoal and current-position coordinates, and
    epsilon is some parameter.

    Note that for this to work, a "minimum activity" threshold must be set, such
    that any activity lower than this is set directly to zero. The threshold is
    the variable --cutoff-- below.

    """
    cutoff = 0.01  # Any dimension smaller than this is set to zero
    total_time = 100  # seconds
    monitoring_interval = 0.1  # seconds
    prob_proceed = 0.01

    signal_strength = 0.011  # Must be bigger than cutoff

    assert cutoff < signal_strength, ('If cutoff is bigger, '
                                      'than signal_strength, '
                                      'the pulse does nothing.')

    num_neurons = 10
    sequence = np.arange(8)
    num_goals = len(sequence)

    epsilon_goal = 0.05  # if (curr_x - goal_x) / goal_x < epislon_goal, goal
                         # achieved

    mash = att.Shs(num_neurons, sequence)
    initial_x = mash.set_initial_conditions(background_noise=0)
    sigma = mash.sigma

    subgoals = np.zeros((num_goals, num_neurons))
    subgoals[np.arange(num_goals), sequence] = sigma[sequence]

    pulses = []  # records pulses
    x_tau = initial_x
    pandata = pd.DataFrame()
    ix_goal = 1  # Indexes the current subgoal (in sequence[])
    for tau in np.arange(0, total_time, monitoring_interval):
        pantau = mash.integrate(t_ini=tau, t_end=tau + monitoring_interval,
                                x_ini=x_tau)
        pandata = pandata.append(pantau)
        x_tau = pantau.iloc[-1].values[:-1]
        distance = np.linalg.norm(x_tau - subgoals[ix_goal, :]).sum() / \
            np.linalg.norm(subgoals[ix_goal, :])
        if distance <= epsilon_goal:
            if np.random.rand() > prob_proceed:
                continue
            malogger.info('Pulse sent at {}'.format(tau + monitoring_interval))
            ix_goal += 1
            if ix_goal >= num_goals:
                break
            x_tau += signal_strength
            pulses.append(tau + monitoring_interval)
        x_tau[x_tau <= cutoff] = 0
    _plot_signal_next(pandata, pulses)
    return pandata, pulses


def _plot_signal_next(pandata, pulses, axis=None):
    """Plots the results of signal_next(). """
    if axis is None:
        axis = plt.subplot(111)
    pandata.plot(x='t', y=pandata.columns[:-1], ax=axis)
    for pulse in pulses:
        axis.plot([pulse, pulse], [-1, 3], color='black')
    plt.show(block=False)


def simulate_two_level(marm=None):
    """IOU
    """
    cutoff = 0.05  # Any dimension smaller than this is set to zero
    total_time = 100  # seconds
    monitoring_interval = 0.1  # seconds
    e_stag = 0.0001

    signal_strength = 0.051  # Must be bigger than cutoff

    assert cutoff < signal_strength, ('If cutoff is bigger, '
                                      'than signal_strength, '
                                      'the pulse does nothing.')

    num_neurons = 5
    sequence = np.arange(5)
    num_goals = len(sequence)

    epsilon_goal = 0.05  # if (curr_x - goal_x) / goal_x < epislon_goal, goal
                         # achieved

    mash = att.Shs(num_neurons, sequence, tau=20)
    initial_x = mash.set_initial_conditions(background_noise=0)
    sigma = mash.sigma

    subgoals = np.zeros((num_goals, num_neurons))
    subgoals[np.arange(num_goals), sequence] = sigma[sequence]

    # Set up the arm
    goals_2d = _goals()
    y_tau = np.array([0, 0])
    # marm = arm.FirstOrderArm(x_ini=y_tau, x_end=goals_2d[1])
    if marm is None:
        marm = two_well_arm(x_ini=y_tau, x_end=goals_2d[1])
    else:
        marm.x_ini = y_tau
        marm.x_end = goals_2d[1]

    # angle_ini = _find_best_angle(marm)
    # initial_speed = np.array([np.cos(angle_ini), np.sin(angle_ini)])
    initial_speed = maximum_likelihood(marm)[0]

    pulse_duration = 0.1

    pulses = []  # records pulses
    x_tau = initial_x
    pandata_x = pd.DataFrame()
    pandata_y = pd.DataFrame()

    # include initial conditions
    pandata_y = pandata_y.append(_pandify_arm(y_tau[:, None], np.array(0)[None, None]))
    ix_goal = 1  # Indexes the current subgoal (in sequence[])
    last_switch = 0
    for tau in np.arange(0, total_time, monitoring_interval):
        pantau = mash.integrate(t_ini=tau, t_end=tau + monitoring_interval,
                                x_ini=x_tau)
        pandata_x = pandata_x.append(pantau)
        x_tau = pantau.iloc[-1].values[:-1]
        if tau - last_switch < pulse_duration:
            tau_y = 0
        else:
            tau_y = tau
        t_interval = (tau_y, tau_y + monitoring_interval)
        marm.x_ini = y_tau
        marm.x_end = np.dot(goals_2d.T, x_tau / np.linalg.norm(x_tau))
        armtau = marm.solve_initial_value(t_interval=t_interval,
                                          initial_speed=initial_speed)
        y_tau = armtau.y[:, -1]
        tau_y = armtau.t[None, :] + tau * (not bool(tau_y))
        pandata_y = pandata_y.append(_pandify_arm(armtau.y, tau_y))
        distance = np.linalg.norm(y_tau - goals_2d[ix_goal, :])
        stagnated = np.linalg.norm(pandata_y.iloc[-1][:-1] - pandata_y.iloc[-2][:-1]) < e_stag
        if distance <= epsilon_goal or stagnated:
            # print('dist:', distance, '\nstag:', stagnated)
            malogger.info('Pulse sent at {}'.format(tau + monitoring_interval))
            ix_goal += 1
            # print(ix_goal, '/', num_goals)
            if ix_goal >= num_goals:
                break
            x_tau += signal_strength
            pulses.append(tau + monitoring_interval)
            marm.x_ini = y_tau
            marm.x_end = goals_2d[ix_goal, :]
            # angle_ini = _find_best_angle(marm)
            # initial_speed = np.array([np.cos(angle_ini), np.sin(angle_ini)])
            initial_speed = maximum_likelihood(marm)[0]
            last_switch = tau
        x_tau[x_tau <= cutoff] = 0
    return pandata_x, pandata_y, pulses, marm


def _goals():
    """Return a list of goal positions in a 2D space.

    """
    return np.array([[0.5, 0], [0.3, 0.8], [1, 0.5], [0.2, 0.1], [0, 0]])


def _find_best_angle(marm):
    samples = arm.infer_best_path(arm=marm)
    return samples.get_values('angle_ini').mean()


def _pandify_arm(y_tau, time):
    """This really does not belong in this file.

    """
    # time = np.array(time, ndmin=2)
    # y_tau = np.array(y_tau, ndmin=2)
    stacked = np.concatenate([y_tau, time], axis=0)
    return pd.DataFrame(stacked.T, columns=['y0', 'y1', 't'])


def two_well_arm(*args, **kwargs):
    """Creates an arm with two wells.

    """
    marm = arm.FirstOrderArm(*args, **kwargs)
    well_1 = {'radius': 0.1, 'center': np.array((0.5, 0.3)),
              'slope': -25 / 0.01, 'alpha': 25}
    well_fun_1 = arm._well_function(**well_1)
    well_2 = {'radius': 0.1, 'center': np.array((0.6, 0.7)),
              'slope': -25 / 0.01, 'alpha': 25}
    well_fun_2 = arm._well_function(**well_2)
    marm.obstacles = [well_1, well_2]

    def funny(x, le_fun=marm.priors_fun, well_fun_1=well_fun_1,
              well_fun_2=well_fun_2):
        return 0 * le_fun(x) + well_fun_1(x) + well_fun_2(x)
    marm.priors_fun = funny
    return marm


def plot_two_level(panda_x, panda_y, pulses, fignum=1):
    """Plots the results from simulate_two_level().

    """
    fig, axes = plt.subplots(2, 1, num=fignum, clear=True, sharex=True)

    _plot_signal_next(panda_x, pulses, axes[0])
    axes[1].plot(panda_y['t'], panda_y.loc[:, ['y0', 'y1']])
    plt.draw()
    plt.show(block=False)


def animate_two_level(panda_x, panda_y, pulses, marm, fignum=2):
    """Creates animations of the results from simulate_two_level

    """
    def data_gen():
        for idr, row in panda_y.iterrows():
            yield row.loc[['y0', 'y1']]

    def init():
        axis.set_ylim(-0.1, 1.1)
        axis.set_xlim(-0.1, 1.1)
        del xdata[:]
        del ydata[:]
        line.set_data(xdata, ydata)
        if hasattr(marm, 'obstacles'):
            circles = [plt.Circle(obstacle['center'], obstacle['radius'])
                       for obstacle in marm.obstacles]
            axis.add_patch(circles[0])
            axis.add_patch(circles[1])
        axis.scatter(*_goals()[1:].T, marker='x', color='orange')
        return line
    
    fig, axis = plt.subplots(1, 1, num=fignum, clear=True)
    line = axis.plot([], [], lw=2)[0]
    axis.grid()
    xdata, ydata = [], []
    # under_axis = fig.add_subplot(111)
    # circles = [plt.Circle(obstacle['center'], obstacle['radius'])
    #            for obstacle in marm.obstacles]

    def run(data):
        t, y = data
        xdata.append(t)
        ydata.append(y)
        xmin, xmax = axis.get_xlim()
        if t >= xmax:
            axis.set_xlim(xmin, 2*xmax)
            axis.figure.canvas.draw()
        line.set_data(xdata, ydata)
        return line
    mani = ani.FuncAnimation(fig, run, data_gen, interval=10, init_func=init)
    plt.show(block=False)
    return mani


def two_simulations():
    """Simulates a free arm and a two-well arm. Same trajectory. Creates
    animations.

    """
    marm1 = arm.FirstOrderArm()
    marm1.priors_fun = lambda x: 0
    marm2 = two_well_arm()

    out1 = simulate_two_level(marm1)
    out2 = simulate_two_level(marm2)

    mani1 = animate_two_level(*out1, fignum=100)
    mani2 = animate_two_level(*out2, fignum=200)

    return mani1, mani2
    return mani2


def test_likelihood(marm=None):
    """Test to see why the fuck the likelihood function isn't doing its
    job

    """
    if marm is None:
        marm = two_well_arm()
        marm.x_ini = np.array([0.5, 0.0])
        marm.x_end = np.array([0.5, 1])

    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    likelihood = np.zeros_like(angles)

    for ix_angle, angle in enumerate(angles):
        initial_speed = np.array([np.cos(angle), np.sin(angle)])
        likelihood[ix_angle] = arm._log_likelihood(marm, initial_speed)
    return likelihood, angles, marm


def maximum_likelihood(marm=None):
    """Finds the best initial speed according to test_likelihood"""
    likelihood, angles, marm = test_likelihood(marm)
    best_angle = angles[np.argmax(likelihood)]
    initial_speed = np.array([np.cos(best_angle), np.sin(best_angle)])
    return initial_speed, marm


def plot_best_likelihood(marm=None, axis=None, fignum=3):
    """Plots the trajectory of the best of test_likelihood()."""
    initial_speed, marm = maximum_likelihood(marm)
    integrated = marm.solve_initial_value(initial_speed=initial_speed)
    if axis is None:
        fig, axis = plt.subplots(1, 1, num=fignum, clear=True, figsize=[4, 4])
    if hasattr(marm, 'obstacles'):
        circles = [plt.Circle(thing['center'], thing['radius'], color='black')
                   for thing in marm.obstacles]
        axis.add_patch(circles[0])
        axis.add_patch(circles[1])
    axis.plot(*integrated.sol(np.linspace(0, 1, 50)))
    axis.scatter(*marm.x_ini)
    axis.text(*marm.x_ini, s='Start')
    axis.scatter(*marm.x_end)
    axis.text(*marm.x_end, s='End')

    axis.axis('equal')
    axis.set(xlim=(-0.1, 1.1), ylim=(-0.1, 1.1))

    plt.draw()
    plt.show(block=False)
    return integrated


def plot_many_examples():
    """Plots several examples of plot_best_likelihood."""
    paths = np.array([[[0, 0], [1, 1]],
                      [[0.5, 0], [0.6, 0.5]],
                      [[0.3, 0.3], [0.7, 0.3]]])
    marm_empty = arm.FirstOrderArm()
    marm_empty.priors_fun = lambda x: 0

    marm_wells = two_well_arm()
    for idx, path in enumerate(paths):
        x_ini, x_end = path
        marm_empty.x_ini = marm_wells.x_ini = x_ini
        marm_empty.x_end = marm_wells.x_end = x_end
        plot_best_likelihood(marm_empty, fignum=10 + idx)
        plot_best_likelihood(marm_wells, fignum=100 + idx)
        
