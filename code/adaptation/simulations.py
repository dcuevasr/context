# -*- coding: utf-8 -*-
# ./adaptation/simulations.py

"""Some simulations for the agent and the task."""

import multiprocessing as mp
from itertools import product
import glob

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import model
import task_hold_hand as thh


def test_obsnoise(fignum=1):
    """Runs the game with increasingly high observation noise, for all three
    agents. Plots the results in a big platter for consumption.

    """
    cue_noise = 0.001
    obs_noises = np.arange(0, 0.3, 0.1)
    fig, axes = plt.subplots(3, len(obs_noises), num=fignum, clear=True,
                             sharex=True, sharey=True)
    for ix_noise, obs_noise in enumerate(obs_noises):
        kwargs = {'angles': [0, 0, 0], 'obs_sd': obs_noise,
                  'cue_noise': cue_noise}
        agents = [model.LeftRightAgent(**kwargs),
                  model.LRMean(**kwargs),
                  model.LRMeanSD(**kwargs)]
        for ix_agent, agent in enumerate(agents):
            thh.run(agent)
            thh.run(agent, cue_key=[0, 2, 1])
            agent.plot_mu(axis=axes[ix_agent, ix_noise])
            agent.plot_position(axis=axes[ix_agent, ix_noise])
            if ix_agent == 0:
                axes[ix_agent, ix_noise].set_title('Obs Noise: {}'.format(obs_noise))
    plt.ylabel('Adaptation')
    plt.xlabel('Trial')
    plt.draw()
    plt.show(block=False)


def test_deadaptation(fignum=2):
    """Runs the game once with some cues and once again with cues that have
    the opposite meaning.

    """
    kwargs = {'angles': [0, 0, 0], 'obs_sd': 1,
              'cue_noise': 0.001}
    fig, axes = plt.subplots(3, 1, num=fignum, clear=True,
                             sharex=True, sharey=True, squeeze=True)
    agents = [model.LeftRightAgent(**kwargs),
              model.LRMean(**kwargs),
              model.LRMeanSD(**kwargs)]
    for ix_agent, agent in enumerate(agents):
        thh.run(agent)
        thh.run(agent, cue_key=[0, 2, 1])
        agent.plot_mu(axis=axes[ix_agent])
        agent.plot_contexts(axis=axes[ix_agent])
        agent.plot_position(axis=axes[ix_agent])
    plt.ylabel('Adaptation')
    plt.xlabel('Trial')
    plt.draw()
    plt.show(block=False)


def grid_sims():
    """Makes the agent(s) perform the task for all the parameter values.
    """
    obs_noises = np.arange(0, 2, 0.1)
    cue_noises = np.arange(0, 0.05, 0.005)

    agents = [model.LeftRightAgent,
              model.LRMean,
              model.LRMeanSD]
    all_pars = product(obs_noises, cue_noises, agents)
    num_pars = len(obs_noises) * len(cue_noises) * len(agents)
    my_robots = mp.Pool()
    num_seeds = num_pars

    seeds = np.random.randint(low=0, high=10000, size=num_seeds * 10)
    seeds = np.unique(seeds)

    for ix_pars, pars in enumerate(all_pars):
        my_robots.apply_async(_one_sim, list(pars))

    my_robots.close()
    my_robots.join()


def _one_sim(obs_noise, cue_noise, agent_fun):
    """Runs one iteration for grid_sims(). """
    agent = agent_fun(obs_sd=obs_noise, cue_noise=cue_noise)
    filename = 'grid_sims_{}_{}_{}.pi'.format(obs_noise, cue_noise, agent.name)
    thh.run(agent=agent, save=True, filename=filename)


def read_data():
    """Reads the data files with the format in _one_sim() and creates
    a panda with it, including the parameter values as columns.

    """
    data_folder = './sim_data/'
    files = glob.iglob(data_folder + 'grid_sims_*.pi')
    pandas = []
    for file in files:
        split = file.split('_')
        panda = pd.read_pickle(file)
        len_panda = len(panda)
        obs_noise = float(split[3]) * np.ones(len_panda)
        cue_noise = float(split[4]) * np.ones(len_panda)
        agent = [split[5][:-3]] * len_panda
        panda['obs_noise'] = obs_noise
        panda['cue_noise'] = cue_noise
        panda['agent'] = agent
        pandas.append(panda)
    pandatron = pd.concat(pandas, axis=0)
    pandatron.reset_index(inplace=True)
    pandatron['index'] = np.arange(len(pandatron))
    pandatron.set_index('index', inplace=True)
    return pandatron


def performance(pandata):
    """Calculates the performance of each agent and each parameter combination in
    --pandata--. Performance is calculated as the sum of the absolute value of hand
    position.

    Returns
    performance : pd.DataFrame

    """
    pandata = pandata.copy()
    pandata['abs_dev'] = pandata['hand'].apply(abs)
    return pandata.groupby(['agent', 'obs_noise', 'cue_noise']).sum()['abs_dev']


def context_inference(pandata):
    """Calculates how well each agent in --pandata-- inferred the context. For every
    trial that the agent mis-identified the context, a 1 is added to the score. Lower
    scores mean better inference.

    """
    pandata = pandata.copy()
    ix_delete = np.array(pandata['ix_context'] == 'clamp')
    pandata.drop(np.nonzero(ix_delete)[0], inplace=True)
    pandata.dropna(axis=0, how='any', inplace=True)
    pandata['context_error'] = pandata['ix_context'].astype(float) != pandata['con_t']
    return pandata.groupby(['agent', 'obs_noise', 'cue_noise']).sum()['context_error']


def interactive_plot(pandata, axis=None, fignum=3):
    """Generator to be able to navigate through the different agents in --pandata--.

    Send an signed int N to go back or forth N agents. Send a list with parameter
    values [obs_noise, cue_noise, agent] to go to that agent.

    """
    if axis is None:
        fig, axis = plt.subplots(1, 1, clear=True, num=fignum)
    plt.show(block=False)
    pandata = pandata.set_index(['obs_noise', 'cue_noise', 'agent'])
    indices = list(pandata.index.unique())
    num_indices = len(indices)
    ix_ix = 0
    while 1:
        pandatum = pandata.loc[indices[ix_ix]]
        num_trials = len(pandatum)
        real_con = np.zeros((num_trials, 4))
        real_con[pandatum['ix_context'] == '0.0', 0] = 1
        real_con[pandatum['ix_context'] == '1.0', 1] = 1
        real_con[pandatum['ix_context'] == '2.0', 2] = 1
        real_con[pandatum['ix_context'] == 'clamp', 3] = 1
        axis.clear()
        axis.plot(np.array(pandatum['hand']), color='black', alpha=0.4)
        axis.plot(np.array(pandatum['mag_mu_0']), color='black')
        axis.plot(np.array(pandatum['mag_mu_1']), color='red')
        axis.plot(np.array(pandatum['mag_mu_2']), color='blue')
        ylim = axis.get_ylim()
        conx = np.array(pandatum.loc[:, ['con0', 'con1', 'con2']])
        axis.plot(ylim[0] + conx[:, 0], color='black')
        axis.plot(ylim[0] + conx[:, 1], color='red')
        axis.plot(ylim[0] + conx[:, 2], color='blue')
        axis.plot(ylim[0] - 1.5 + real_con[:, 0], color='black')
        axis.plot(ylim[0] - 1.5 + real_con[:, 1], color='red')
        axis.plot(ylim[0] - 1.5 + real_con[:, 2], color='blue')
        axis.plot(ylim[0] - 1.5 + real_con[:, 3], color='green')
        axis.set_title(indices[ix_ix])
        plt.draw()
        input = yield None
        if input == 0:
            return
        ix_ix += input
        ix_ix = ix_ix % num_indices
