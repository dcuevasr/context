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
from pars import task as task_pars


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
                axes[ix_agent, ix_noise].set_title(
                    'Obs Noise: {}'.format(obs_noise))
    plt.ylabel('Adaptation')
    plt.xlabel('Trial')
    plt.draw()
    plt.show(block=False)


def test_deadaptation(fignum=2):
    """Runs the game once with some cues and once again with cues that have
    the opposite meaning.

    """
    kwargs = {'angles': [0, 0, 0], 'obs_sd': task_pars['obs_noise'],
              'context_noise': 0.1,
              'cue_noise': 0.01}
    fig, axes = plt.subplots(3, 1, num=fignum, clear=True,
                             sharex=True, sharey=True, squeeze=True)
    agents = [model.LeftRightAgent(**kwargs),
              model.LRMean(**kwargs),
              model.LRMeanSD(**kwargs)]
    for ix_agent, agent in enumerate(agents):
        thh.run(agent)
        agent.plot_mu(axis=axes[ix_agent])
        agent.plot_contexts(axis=axes[ix_agent])
        agent.plot_position(axis=axes[ix_agent])
    plt.ylabel('Adaptation')
    plt.xlabel('Trial')
    plt.draw()
    plt.show(block=False)


def grid_sims():
    """Makes the agent(s) perform the task for all the parameter values.

    All your processors are belong to us.

    Outputs are saved to pickled files
    """
    labels, grid_elements = _define_grid()
    labels = labels + ['agent']
    all_pars = product(*grid_elements)
    my_robots = mp.Pool()

    for pars in all_pars:
        inputs = {label: value
                  for label, value in zip(labels, pars)}
        # _one_sim(inputs)
        my_robots.apply_async(_one_sim, args=[inputs])
    my_robots.close()
    my_robots.join()


def _define_grid():
    """Defines the grid for grid_sims and read_data.  This function is meant to be
    edited manually to do different simulations. Keep in mind that the labels
    must match those expected by the agents' instantiation functions.

    """
    from sys import version_info
    if version_info[0] < 3 or version_info[1] < 7:
        raise NotImplementedError('The functions that use this function '
                                  'rely on dictionaries being ordered and '
                                  'I was too lazy to implement it, so '
                                  'python 3.7+ is required.')
    pred_noises = np.arange(0, 0.2, 0.03)
    cue_noises = np.arange(0, 0.05, 0.005)
    con_noises = np.arange(0, 0.3, 0.1)
    force_sds = np.tile(np.arange(0.0001, 0.002, 0.0002), (3, 1)).T

    agents = [model.LeftRightAgent,
              model.LRMean,
              model.LRMeanSD]
    values = [pred_noises, cue_noises, con_noises, force_sds]
    labels = ['prediction_noise', 'cue_noise', 'context_noise', 'force_sds']
    try:  # Test labels on agent
        test_values = [value[0] for value in values]
        agents[0](**{label: value for label, value in zip(labels, test_values)})
    except TypeError:
        raise ValueError('Testing the labels on the agent failed. '
                         'Maybe the labels are misspelled?')
    values.append(agents)
    return labels, values


def _one_sim(kwargs):
    """Runs one iteration for grid_sims(). """
    from pars import task as task_pars
    agent = kwargs.pop('agent')
    agent = agent(**kwargs, angles=[0, 0, 0])
    file_stem = 'grid_sims' + '_{}' * (len(kwargs) + 1) + '.pi'
    values = []
    for value in kwargs.values():
        if np.size(value) > 1:
            c_value = value[0]
        else:
            c_value = value
        values.append(c_value)
    filename = file_stem.format(*values, agent.name)
    thh.run(agent=agent, save=True, filename=filename, pars=task_pars)


def read_data():
    """Reads the data files with the format in _one_sim() and creates
    a panda with it, including the parameter values as columns.

    """
    labels, _ = _define_grid()
    data_folder = './sim_data/'
    files = glob.iglob(data_folder + 'grid_sims_*.pi')
    pandas = []
    for file in files:
        split = file.split('_')[3:]
        panda = pd.read_pickle(file)
        len_panda = len(panda)
        for ix_part, part in enumerate(split[:-1]):
            split_split = part.split(' ')
            if len(split_split) > 1:
                c_part = split_split[0][1:]
            else:
                c_part = part
            panda[labels[ix_part]] = float(c_part) * np.ones(len_panda)
        # cue_noise = float(split[3]) * np.ones(len_panda)
        # con_noise = float(split[4]) * np.ones(len_panda)
        panda['agent'] = [split[-1][:-3]] * len_panda
        # panda['cue_noise'] = cue_noise
        # panda['context_noise'] = con_noise
        # panda['agent'] = agent
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
    labels, _ = _define_grid()
    return pandata.groupby(labels).sum()['abs_dev']


def context_inference(pandata):
    """Calculates how well each agent in --pandata-- inferred the context. For
    every trial that the agent mis-identified the context, a 1 is added to the
    score. Lower scores mean better inference.

    """
    pandata = pandata.copy()
    ix_delete = np.array(pandata['ix_context'] == 'clamp')
    pandata.drop(np.nonzero(ix_delete)[0], inplace=True)
    pandata.dropna(axis=0, how='any', inplace=True)
    pandata['context_error'] = pandata['ix_context'].astype(float) != pandata['con_t']
    labels, _ = _define_grid()
    return pandata.groupby(labels).sum()['context_error']


def interactive_plot(pandata, axis=None, fignum=3):
    """Generator to be able to navigate through the different agents in
    --pandata--.

    Send an signed int N to go back or forth N agents. Send a list with
    parameter values [obs_noise, cue_noise, agent] to go to that agent.

    """
    labels, _ = _define_grid()
    if axis is None:
        fig, axis = plt.subplots(1, 1, clear=True, num=fignum)
    plt.show(block=False)
    pandata = pandata.set_index(labels + ['agent'])
    indices = list(pandata.index.unique())
    num_indices = len(indices)
    ix_ix = 0
    while 1:
        pandatum = pandata.loc[indices[ix_ix]]
        num_trials = len(pandatum)
        real_con = np.zeros((num_trials, 4))
        axis.clear()
        axis.plot(np.array(pandatum['pos(t)']), color='black', alpha=0.4)
        axis.plot(np.array(pandatum['mag_mu_0']), color='black')
        axis.plot(np.array(pandatum['mag_mu_1']), color='red')
        axis.plot(np.array(pandatum['mag_mu_2']), color='blue')
        ylim = axis.get_ylim()
        height = np.abs(np.max(ylim) - np.min(ylim)) * 0.2
        real_con[pandatum['ix_context'] == 0, 0] = height
        real_con[pandatum['ix_context'] == 1, 1] = height
        real_con[pandatum['ix_context'] == 2, 2] = height
        real_con[pandatum['ix_context'] == 'clamp', 3] = height
        conx = np.array(pandatum.loc[:, ['con0', 'con1', 'con2']]) * height
        axis.plot(ylim[0] + conx[:, 0], color='black')
        axis.plot(ylim[0] + conx[:, 1], color='red')
        axis.plot(ylim[0] + conx[:, 2], color='blue')
        axis.plot(ylim[0] - 1.5 * height + real_con[:, 0], color='black')
        axis.plot(ylim[0] - 1.5 * height + real_con[:, 1], color='red')
        axis.plot(ylim[0] - 1.5 * height + real_con[:, 2], color='blue')
        axis.plot(ylim[0] - 1.5 * height + real_con[:, 3], color='green')
        axis.plot(np.array(pandatum['action']), color='yellow')
        # axis.plot(np.array(pandatum['action']), color='yellow')
        title = ''.join([label + ': {}\n' for label in labels]) + 'Agent: {}\n'
        axis.set_title(title.format(*indices[ix_ix]))
        plt.draw()
        input = yield None
        if input == 0:
            fig.close()
            yield None
            return
        ix_ix += input
        ix_ix = ix_ix % num_indices


if __name__ == '__main__':
    grid_sims()
