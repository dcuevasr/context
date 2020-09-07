# -*- coding: utf-8 -*-
# ./hand_gestures.py


"""Everything hand gesture-related goes here. Not sure what that will be."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import attractors as att

NUM_DIM = 15  # 5 fingers, 3D space


def define_gestures():
    """Defines all the gestures available to a system in terms of equilibrium
    points in the 5x3-dimensional phase system of fingertip positions.

    The coordinate system works as follows: each fingertip has a 3D vector
    associated with its position. The dimensions are width (thumb to pinky),
    height (bottom of palm to fingertip of middle finger) and width (from back
    of palm to as far forward as the fingers can point away from the palm). The
    vector is normalized, so all dimensions move from 0 to 1.

    The order of the vectors is thumb->pinky.

    Returns
    -------
    gestures : dict
    Dictionary whose keys are gesture names (e.g. "pinch") and the values are
    the 5-dimensional position of an equilibrium point in phase space of the
    fingers of one hand. Note that the equilibrium points are returned as a 5x3
    array, which may need to be flattened before use.

    """
    gestures = {}

    gestures['pinch'] = np.array([[0.2, 0.5, 0.8],
                                  [0.2, 0.5, 0.8],
                                  [0.5, 0.6, 0.3],
                                  [0.7, 0.7, 0.3],
                                  [0.9, 0.8, 0.1]])

    gestures['tube'] = np.array([[0.2, 0.3, 0.9],
                                 [0.3, 0.5, 0.8],
                                 [0.5, 0.6, 0.8],
                                 [0.7, 0.5, 0.8],
                                 [0.9, 0.4, 0.8]])

    gestures['ball'] = np.array([[0.05, 0.2, 0.1],
                                 [0.25, 0.8, 0.1],
                                 [0.45, 0.9, 0.1],
                                 [0.65, 0.8, 0.1],
                                 [1.0, 0.6, 0.1]])

    return gestures


def example_switching():
    """Shows the switching behavior of the system: going from one hand gesture
    to another to another.

    The gestures are taken directly from define_gestures().

    """
    num_gestures = 10
    num_dims = NUM_DIM
    time_per_gesture = 5

    gestures = define_gestures()

    gesture_names = list(gestures.keys())

    # seq_gestures = np.random.choice(gesture_names, replace=True,
    #                                 size=num_gestures)
    seq_gestures = np.repeat(gesture_names, 5)
    matt = att.PointAttractor(pos=np.zeros(num_dims), eta=-np.ones(num_dims))
    c_pos = gestures[seq_gestures[-1]].reshape(-1)
    pandata = pd.DataFrame()
    for ix_gesture in range(num_gestures - 1):
        t_ini = ix_gesture * time_per_gesture
        t_end = (ix_gesture + 1) * time_per_gesture
        matt.set_pos(gestures[seq_gestures[ix_gesture + 1]].reshape(-1))
        integrate_out = matt.integrate(t_end=t_end, t_ini=t_ini, x_ini=c_pos)
        pandata = pandata.append(integrate_out)
        c_pos = pandata.iloc[-1][:-1]
    return pandata


def plot_hands(positions, fignum=2):
    """Plots the hands' positions in --positions--.

    Parameters
    ----------
    positions : 1darray, 2darray or DataFrame
    Positions of the hands to plot. If 1darray, it is assumed to be a flattened
    3x5 array (ValueError if not) with a single time point to plot. If 2darray,
    it is assumed to be the (5, 3) array of positions for all fingers. If pandas
    DataFrame, it is assumed that it has 15 spatial columns (named x0 to x14)
    and the column t with the time points; it's the format of
    att.BaseAttractor.integrate().

    """
    if isinstance(positions, np.ndarray):
        positions = positions.reshape(-1)
        if positions.size != NUM_DIM:
            raise ValueError('Number of spatial dimensions in --positions-- '
                             'differs from {}'.format(NUM_DIM))
        positions = np.concatenate([positions, 0])
        positions = pd.DataFrame(positions,
                                 columns=[*['x{}'.format(idx)
                                            for idx in range(NUM_DIM)], 't'])
    fig = plt.figure(num=fignum, clear=True)
    axis = fig.add_subplot(111, projection='3d')
    plt.show(block=False)
    for row in range(len(positions)):
        axis.clear()
        plot_one_hand(positions.iloc[row].values[:-1], axis=axis)
        axis.set_xlim((0, 1))
        axis.set_ylim((0, 1))
        axis.set_zlim((0, 1))
        plt.pause(0.2)
        plt.draw()


def plot_one_hand(positions, fignum=1, axis=None, offset=np.zeros(3)):
    """Draws a single hand given the fingertip positions in --positions--.

    Parameters
    ----------

    positions : 1darray, 2darray
    Positions of the hands to plot. If 1darray, it is assumed to be a flattened
    3x5 array (ValueError if not) with a single time point to plot. If 2darray,
    it is assumed to be the (5, 3) array of positions for all fingers.

    fignum : int
    Number of the figure to create. Defaults to 1. If --axis-- is provided, this
    is ignored.

    axis : Axis class from matplotlib
    Axis to draw the hand. If None, a new figure is created.

    offset : 1darray
    x, y, z coordinates of the offset. Used to draw the hand anywhere
    other than the center of the axis.

    Returns
    -------
    fig, axis
    Only if --axis-- was not provided (None).

    """
    positions = positions.reshape((5, 3))
    palm_center = np.array([0.5, 0.5, 0]) + offset
    flag_return_axis = False
    if axis is None:
        flag_return_axis = True
        fig = plt.figure(num=fignum, clear=True)
        axis = fig.add_subplot(111, projection='3d')
        plt.show(block=False)

    for finger in positions:
        values = list(zip(palm_center, finger))
        axis.plot(*values)

    if flag_return_axis:
        return fig, axis
