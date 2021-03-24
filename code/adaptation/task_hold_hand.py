# -*- coding: utf-8 -*-
# ./adaptation/task_hold_hand.py
# import ipdb
from datetime import datetime

import numpy as np
from scipy import stats
import pandas as pd

import model

"""Task described in section 7.1 of notes.pdf in which the participant
must hold her hand in the starting position in different force fields.
The game is meant to be played by model.LeftRightAgent or its children.

"""


def run(agent=None, save=False, filename=None, pars=None):
    """Runs an entire game of the holding-hand task."""
    if pars is None:
        from pars import pars
    if agent is None:
        agent = model.LeftRightAgent(obs_sd=pars['obs_noise'][1])
    outs = []
    for ix_miniblock, _ in enumerate(pars['context_seq']):
        out = miniblock(ix_miniblock, 0, agent, pars)
        for one_out in out:
            outs.append(one_out)
    pandata = pd.DataFrame(outs,
                           columns=('action', 'force', 'pos(t)', 'pos(t+1)',
                                    'ix_context'))
    pandata.rename_axis('trial', inplace=True)
    pandagent = agent.pandify_data()
    if save:
        save_pandas(pandata, pandagent, filename)
    return pandata, pandagent, agent


def miniblock(ix_context, hand_position, agent, pars):
    """Runs a miniblock of the task."""
    outs = []
    for ix_trial in range(pars['num_trials']):
        out = trial(ix_context, hand_position, agent, pars)
        outs.append(out)
        hand_position = out[3]
    # outs = np.stack(outs, axis=0)
    return outs


def trial(ix_miniblock, hand_position, agent, pars):
    """Runs a single trial of the task.

    Parameters
    ----------
    ix_context : int or string
    Index that indicates the current context. Should index everything
    context-related in the configuration dictionary --pars--. If 'clamp',
    it is taken as an error-clamp trial, in which the position of the
    hand is held at zero regardless of the action.

    hand_position : 1darray or float
    Current position in Cartesian coordinates. Can be a float in the
    case of one-dimensional spaces. Note that (0, 0, ...) is both
    the origin and the starting position of the hand.

    agent : model.LeftRightAgent instance
    Agent that makes decisions. Needs one_trial(hand_position, cue) method as
    well as log_context attribute.

    cue_key : iterable
    Iterable to fake cues. If in a trial the context is 2, cue_key[2] will
    be used. If cue_key=[0, 1, 2] (the default), this has no effect. If
    something else, the agent will be lied to. LIES!

    """
    context = pars['context_seq'][ix_miniblock]
    cue = pars['cues'][ix_miniblock]
    c_hand_position = hand_position
    c_obs = sample_observation(hand_position, pars)
    action = agent.one_trial(c_obs, cue=cue)
    if context == 'clamp':
        n_hand_position = 0
        force = -action
    else:
        force = sample_force(context, pars)
        n_hand_position = c_hand_position + force + action
    outs = [action, force, c_hand_position, n_hand_position,
            context]
    return outs


def sample_observation(hand_position, pars):
    """Generates an observation given the current position and
    the context.

    TODO: Maybe turn into a generator, so the normal doesn't
          have to be instantiated every time.

    """
    try:
        if len(hand_position) > 1:
            raise NotImplementedError('Havent implemented ' +
                                      'n-dimensional spaces.')
    except TypeError:
        pass
    loc, scale = pars['obs_noise']
    loc += hand_position
    distri = stats.norm(loc=loc, scale=scale)
    return distri.rvs()


def sample_force(context, pars):
    """Samples the force exerted by the environment on the force
    given the --hand_position-- and the context.

    """
    loc, scale = pars['force_noise'][context]
    distri = stats.norm(loc=loc, scale=scale)
    magnitude = np.prod(pars['forces'][context])
    return magnitude + distri.rvs()


def join_pandas(pandata, pandagent):
    """Joins the task data and the agent's data (outputs of run())
    into one big panda, aligned by trial number. Returns the panda.

    """
    pandagent = pandagent.drop(['action', 'hand'], axis=1)
    pandagent = pandagent.iloc[pandagent.index != -1]
    return pd.concat([pandata, pandagent], axis=1)


def save_pandas(pandata, pandagent, filename=None):
    """Saves the pandas to a file with the current date and time
    as the name.

    """
    foldername = './sim_data/'
    if filename is None:
        date = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = 'data_{}.pi'.format(date)
    pandatron = join_pandas(pandata, pandagent)
    pandatron.to_pickle(foldername + filename)


if __name__ == '__main__':
    pandata, pandagent, _ = run()
    save_pandas(pandata, pandagent)
