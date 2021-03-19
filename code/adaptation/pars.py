pars = {'obs_noise': [0, 0.1],  # Observation noise N(mu, sd)
        'force_noise': [[0, 0.0001], [0, 0.2], [0, 0.2]],  # Noise of the force process
        'forces': [[0, 0], [1, 2], [-1, 2]],  # [angle, magnitude] for each force
        'num_trials': 300,  # Trials per miniblock
        'context_seq': [0, 1, 2, 'clamp', 0]  # Sequence of context type to use.
        }
