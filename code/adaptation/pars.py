# -*- coding: utf-8 -*-

import numpy as np

"""Parameters for the model and the task. See /notes/notes.pdf for an
explanation of each one of them and how these values were found.

All values are in the international system of units (m, g, s, N, ...).
"""
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
        'num_trials': 50,  # Trials per miniblock
        'context_seq': [0, 1, 2, 'clamp', 0, 1, 2, 0],  # Sequence of context type to use.
        'cues': [0, 1, 2, 0, 0, 2, 1, 0]
        }

# Default values for the parameters of the agents in models.py
agent = {'obs_noise': obs_noise,
         'delta_t': delta_t,
         'max_force': 2 * max(fake_mags),
         'action_sd': max(fake_mags) / 5,
         'force_sd': force_sd * np.ones(3) * 3
         }
