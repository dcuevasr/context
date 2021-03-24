pars = {'obs_noise': [0, 0.1],  # Observation noise N(mu, sd)
        'force_noise': [[0, 0.0001], [0, 0.2], [0, 0.2]],  # Noise of the force process
        'forces': [[0, 0], [1, 20], [-1, 20]],  # [angle, magnitude] for each force
        'num_trials': 50,  # Trials per miniblock
        'context_seq': [0, 1, 2, 'clamp', 0, 1, 2, 0],  # Sequence of context type to use.
        'cues': [0, 1, 2, 0, 0, 2, 1, 0]
        }
if len(pars['context_seq']) != len(pars['cues']):
    raise ValueError('Number of cues and number of miniblocks do not match.')
