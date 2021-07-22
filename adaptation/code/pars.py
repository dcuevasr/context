# -*- coding: utf-8 -*-

import numpy as np

"""Parameters for the model and the task. See /notes/notes.pdf for an
explanation of each one of them and how these values were found.

All values are in the international system of units (m, g, s, N, ...).
"""

CLAMP_INDEX = 150


def _populate(nonzeros, vector=None, size=None, element=1):
    """Creates a vector of zeros of size --size-- and puts an --element-- in each
    of the indices in --nonzeros--.

    If --vector-- is provided, it is used instead of creating zeros.

    Beware of automatic casting!

    """
    if vector is None:
        vector = np.zeros(size, dtype=int)
    for index in nonzeros:
        vector[int(index)] = element
    return vector


def _define_contexts_base():
    """Defines the context of each trial. This is meant to be edited by
    hand. """
    # baseline = np.arange(50)
    first_adaptation = np.arange(50, 200)
    second_adaptation = np.arange(200, 350)
    deadaptation = np.arange(350, 400)
    clamp = np.arange(400, 500)
    vector = _populate(size=500, nonzeros=first_adaptation, element=1)
    _populate(vector=vector, nonzeros=second_adaptation, element=2)
    _populate(vector=vector, nonzeros=deadaptation, element=2)
    _populate(vector=vector, nonzeros=clamp, element=CLAMP_INDEX)
    return vector


def _define_cues_base():
    """Defines the cues of each trial. This is meant to be edited by hand. """
    # baseline = np.arange(50)
    first_adaptation = np.arange(50, 200)
    second_adaptation = np.arange(200, 350)
    deadaptation = np.arange(350, 400)
    clamp = np.arange(400, 500)
    vector = _populate(size=500, nonzeros=first_adaptation, element=1)
    _populate(vector=vector, nonzeros=second_adaptation, element=2)
    _populate(vector=vector, nonzeros=deadaptation, element=1)
    _populate(vector=vector, nonzeros=clamp, element=1)
    return vector


def _define_breaks():
    """Defines the agent breaks for the entire session. This is meant to be edited
    by hand."""
    breaks = [50, 200, 350, 400]
    vector = _populate(size=500, nonzeros=breaks, element=1)
    return vector


def define_contexts(structure):
    """Given a list with elements [context_id, num_trials], it will piece together
    the entire sequence of trial types for the experiment.

    Example:
    structure_dict = [[0, 10], [1, 20], [0, 5], [2, 100]]
    This will create an experiment with 10 + 20 + 5 + 100 = 135 trials, the
    first 10 being of context 0, the next 20 with context 1, etc.

    Returns
    -------
    contexts : ndarray size=(total_trials, )
    Context for each trial in the experiment.

    """
    context_list = []
    for block in structure:
        context_list.append(block[0] * np.ones(block[1], dtype=int))
    return np.concatenate(context_list)


# Parameters common to agent and task:
delta_t = 0.05
obs_noise = 0.0001
forces = np.array([0, 10, 10])
fake_mags = 0.5 * forces * delta_t ** 2  # Really, see the notes.pdf
force_sd = 0.0001


# Values for the parameters of the task in task_hold_hand.py
task = {'obs_noise': obs_noise,  # Observation noise N(mu, sd)
        'force_noise': force_sd * np.ones(3),  # Noise of the force process
        'forces': [[0, 0], [1, fake_mags[1]], [-1, fake_mags[2]]],  # [angle, magnitude]
        'context_seq': _define_contexts_base(),  # Sequence of context type to use.
        'cues': _define_cues_base(),
        'breaks': _define_breaks(),
        'clamp_index': CLAMP_INDEX
        }

# Default values for the parameters of the agents in models.py
agent = {'obs_noise': obs_noise,
         'delta_t': delta_t,
         'max_force': 1.2 * max(fake_mags),
         'action_sd': max(fake_mags) / 10,
         'force_sd': 2 * force_sd * np.ones(3),
         'prediction_noise': 0.01,
         'reset_after_change': True,  # Whether to reset priors on new miniblock
         }

# The following are examples for different experiments. Maybe move to another file:

# The following parameters follow the experiment in Smith_2006
task_smith = task.copy()
task_smith['num_trials'] = 10
task_smith['context_seq'] = [0] * 12 + [1] * 40 + [2] * 2 + ['clamp'] * 18
task_smith['cues'] = [0] * 12 + [1] * 40 + [1] * 2 + [1] * 18
task_smith['reset_after_change'] = True


# O-A experiment example. No error clamp
num_trials = 570
task_oa = task.copy()
task_oa['obs_noise'] = 0.001
task_oa['num_trials'] = 10
task_oa['context_seq'] = _populate(size=num_trials,
                                   nonzeros=np.arange(120, 520))
task_oa['cues'] = _populate(size=num_trials, nonzeros=np.arange(120, 570))
task_oa['reset_after_change'] = True
task_oa['breaks'] = _populate(size=num_trials, nonzeros=[120, 520],
                              element=1)


# 0-A-0-A-0-A-O-A example. Good cues. Breaks after each block.
num_trials = 500
task_oaoa = task.copy()
task_oaoa['obs_noise'] = 0.001
task_oaoa['num_trials'] = 10
context_seq = _populate(size=num_trials, nonzeros=np.arange(30, 200))
_populate(nonzeros=np.arange(250, 300), vector=context_seq)
_populate(nonzeros=np.arange(350, 400), vector=context_seq)
_populate(nonzeros=np.arange(450, 500), vector=context_seq)
task_oaoa['context_seq'] = context_seq
task_oaoa['cues'] = context_seq
task_oaoa['reset_after_change'] = True
task_oaoa['breaks'] = _populate(size=num_trials,
                                nonzeros=[30, 200, 250, 300, 350, 400, 450],
                                element=1)

# 0-A-0-A-0-A-O-A example. Good cues. No breaks
task_oaoa_nobreak = task_oaoa.copy()
task_oaoa_nobreak['breaks'] = np.zeros_like(task_oaoa_nobreak['context_seq'])


# Baseline bias example following Davidson_Scaling_2004, experiment 1, group 1
num_trials = 320 + 160
task_davidson_1_1 = task.copy()
context_seq = _populate(size=num_trials, nonzeros=np.arange(320))
_populate(vector=context_seq, nonzeros=np.arange(320, num_trials), element=2)
task_davidson_1_1['context_seq'] = context_seq
task_davidson_1_1['cues'] = context_seq
task_davidson_1_1['breaks'] = np.zeros(num_trials)
davidson_1_1_forces = np.array([0, 12, 4])
davidson_1_1_fakes = 0.5 * davidson_1_1_forces * delta_t ** 2
task_davidson_1_1['forces'] = [[0, 0], [1, davidson_1_1_fakes[1]],
                               [1, davidson_1_1_fakes[2]]]


# Baseline bias example following Davidson_Scaling_2004, experiment 1, group 2
num_trials = 320 + 160
task_davidson_1_2 = task.copy()
context_seq = _populate(size=num_trials, nonzeros=np.arange(320))
task_davidson_1_2['context_seq'] = context_seq
task_davidson_1_2['cues'] = context_seq
task_davidson_1_2['breaks'] = np.zeros(num_trials)
davidson_1_2_forces = np.array([0, 12, 4])
davidson_1_2_fakes = 0.5 * davidson_1_2_forces * delta_t ** 2
task_davidson_1_2['forces'] = [[0, 0], [1, davidson_1_2_fakes[1]]]


# Baseline bias example following Davidson_Scaling_2004, experiment 2, group 1
num_trials = 160 * 4
task_davidson_2_1 = task.copy()
context_seq = 2 * np.ones(num_trials, dtype=int)
_populate(vector=context_seq, nonzeros=np.arange(160))
_populate(vector=context_seq, nonzeros=np.arange(320, 480))
task_davidson_2_1['context_seq'] = context_seq
task_davidson_2_1['cues'] = context_seq
task_davidson_2_1['breaks'] = np.zeros(num_trials)
davidson_2_1_forces = np.array([0, 4, 12])
davidson_2_1_fakes = 0.5 * davidson_2_1_forces * delta_t ** 2
task_davidson_2_1['forces'] = [[0, 0], [1, davidson_2_1_fakes[1]],
                               [1, davidson_2_1_fakes[2]]]

# Baseline bias example following Davidson_Scaling_2004, experiment 2, group 2
num_trials = 160 * 4
task_davidson_2_2 = task.copy()
context_seq = 2 * np.ones(num_trials, dtype=int)
_populate(vector=context_seq, nonzeros=np.arange(160))
_populate(vector=context_seq, nonzeros=np.arange(320, 480))
task_davidson_2_2['context_seq'] = context_seq
task_davidson_2_2['cues'] = context_seq
task_davidson_2_2['breaks'] = np.zeros(num_trials)
davidson_2_2_forces = np.array([0, 4, -4])
davidson_2_2_fakes = 0.5 * davidson_2_2_forces * delta_t ** 2
task_davidson_2_2['forces'] = [[0, 0], [1, davidson_2_2_fakes[1]],
                               [1, davidson_2_2_fakes[2]]]
