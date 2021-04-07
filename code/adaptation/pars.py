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

# The following parameters follow the experiment in Smith_2006
task_smith = task.copy()
task_smith['num_trials'] = 10
task_smith['context_seq'] = [0] * 12 + [1] * 40 + [2] * 2 + ['clamp'] * 18
task_smith['cues'] = [0] * 12 + [1] * 40 + [1] * 2 + [1] * 18
task_smith['reset_after_change'] = True
