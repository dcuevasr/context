# -*- coding: utf-8 -*-
# ./adaptation/task_hold_hand.py
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
        from pars import task as pars
    if agent is None:
        agent = model.LeftRightAgent(obs_sd=pars['obs_noise'])
    outs = []
    hand_position = 0
    for c_trial, do_break in enumerate(pars['breaks']):
        # if c_trial >= 111:
        #     ipdb.set_trace()
        if do_break:
            agent.reset()
            hand_position = 0
        out = trial(c_trial, hand_position, agent, pars)
        outs.append(out)
        hand_position = out[3]
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
    agent.reset()
    for ix_trial in range(pars['num_trials']):
        out = trial(ix_context, hand_position, agent, pars)
        outs.append(out)
        hand_position = out[3]
    return outs


def trial(c_trial, hand_position, agent, pars):
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

    """
    context = pars['context_seq'][c_trial]
    cue = pars['cues'][c_trial]
    c_hand_position = hand_position
    c_obs = sample_observation(hand_position, pars)
    action = agent.one_trial(c_obs, cue=cue)
    # agent.cue_history.append(cue)
    # agent.hand_position = c_hand_position
    # agent.hand_position_history.append(c_obs)
    # p_con = agent.log_context
    # agent.infer_context(None, cue)
    # # agent.update_magnitudes()
    # action = agent.make_decision()
    # agent.is_reset = False
    if context == pars['clamp_index']:
        n_hand_position = 0
        force = -action
    else:
        force = sample_force(context, pars)
        n_hand_position = c_hand_position + force + action
    # agent.hand_position = n_hand_position
    # agent.log_context = p_con
    # # del agent.log_context_history[-1]
    # agent.infer_context(n_hand_position, cue=cue, rewrite=True)
    # agent.update_magnitudes()
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
    scale = pars['obs_noise']
    if scale == 0:
        return hand_position
    loc = hand_position
    distri = stats.norm(loc=loc, scale=scale)
    return distri.rvs()


def sample_force(context, pars):
    """Samples the force exerted by the environment on the force
    given the --hand_position-- and the context.

    """
    scale = pars['force_noise'][context]
    distri = stats.norm(loc=0, scale=scale)
    magnitude = np.prod(pars['forces'][context])
    return magnitude + distri.rvs()


def join_pandas(pandata, pandagent):
    """Joins the task data and the agent's data (outputs of run())
    into one big panda, aligned by trial number. Returns the panda.

    """
    pandagent = pandagent.drop(['action'], axis=1)
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
