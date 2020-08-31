# -*- coding: utf-8 -*-
# ../simulations.py


"""Collections of simulations "sin ton ni son"."""

import logging

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import attractors as att


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


def _plot_signal_next(pandata, pulses):
    """Plots the results of signal_next(). """
    pandata.plot(x='t', y=pandata.columns[:-1])
    for pulse in pulses:
        plt.plot([pulse, pulse], [-1, 3], color='black')
    plt.show(block=False)
