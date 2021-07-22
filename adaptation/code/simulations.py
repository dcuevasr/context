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
import pars


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

    for c_pars in all_pars:
        inputs = {label: value
                  for label, value in zip(labels, c_pars)}
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
        agents[0](**{label: value
                     for label, value in zip(labels, test_values)})
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
        panda['agent'] = [split[-1][:-3]] * len_panda
        pandas.append(panda)
    pandatron = pd.concat(pandas, axis=0)
    pandatron.reset_index(inplace=True)
    pandatron['index'] = np.arange(len(pandatron))
    pandatron.set_index('index', inplace=True)
    return pandatron


def performance(pandata):
    """Calculates the performance of each agent and each parameter combination in
    --pandata--. Performance is calculated as the sum of the absolute value of
    hand position.

    Returns
    performance : pd.DataFrame

    """
    pandata = pandata.copy()
    pandata['abs_dev'] = pandata['pos(t)'].apply(abs)
    labels, _ = _define_grid()
    return pandata.groupby(labels + ['agent']).sum()['abs_dev']


def context_inference(pandata):
    """Calculates how well each agent in --pandata-- inferred the context. For
    every trial that the agent mis-identified the context, a 1 is added to the
    score. Lower scores mean better inference.

    """
    pandata = pandata.copy()
    # pandata.dropna(axis=0, how='any', inplace=True)
    pandata['context_error'] = pandata['ix_context'].astype(float) != pandata['con_t']
    labels, _ = _define_grid()
    return pandata.groupby(labels + ['agent']).sum()['context_error']


def scores(pandata):
    """Convenience function to call performance() and context_inference().

    Returns all in a concatenated panda.

    """
    perfy = performance(pandata)
    conty = context_inference(pandata)
    return pd.concat([perfy, conty], axis=1)


def interactive_plot(pandata, axes=None, fignum=3):
    """Generator to be able to navigate through the different agents in
    --pandata--.

    Send an signed int N to go back or forth N agents. Send a list with
    parameter values [obs_noise, cue_noise, agent] to go to that agent.

    """
    labels, _ = _define_grid()
    if axes is None:
        fig, axes = plt.subplots(2, 1, clear=True, num=fignum)
    else:
        if len(axes) != 2:
            error_text = 'Number of axes provided is {}, should be {}.'
            raise ValueError(error_text.format(len(axes), 2))
        plt.show(block=False)
    flag_noindex = False
    try:
        pandata = pandata.set_index(labels + ['agent'])
    except KeyError:
        print('Warning: the given panda does not have at least some '
              'of the columns in _define_grid(); using the indices '
              'present in the panda instead.')
        flag_noindex = True
    indices = list(pandata.index.unique())
    num_indices = len(indices)
    ix_ix = 0
    while 1:
        pandatum = pandata.loc[indices[ix_ix]]
        axes[0].clear()
        axes[1].clear()
        plot_adaptation(pandatum, axis=axes[0])
        plot_contexts(pandatum, axis=axes[1])
        if flag_noindex:
            axes[0].set_title(indices[ix_ix])
        else:
            title = ''.join([label + ': {}\n' for label in labels]) + 'Agent: {}\n'
            axes[0].set_title(title.format(*indices[ix_ix]))
        plt.draw()
        input = yield None
        if input == 0:
            fig.close()
            yield None
            return
        ix_ix += input
        ix_ix = ix_ix % num_indices


def plot_contexts(pandata, axis=None, fignum=4):
    """Plots the inferred contexts as well as the true contexts. True contexts
    are plotted as background colors and the posterior over contexts as
    colored lines. The chromatic code is the same for both, but the alpha on
    the true contexts is lower for visual clarity.

    Parameters
    ----------
    pandata : DataFrame
    Data from both the agent and the task, i.e. the output of thh.join_pandas.

    """
    flag_makepretty = False
    if axis is None:
        fig, axis = plt.subplots(1, 1, clear=True, num=fignum)
        flag_makepretty = True
    else:
        plt.show(block=False)
    alpha = 0.1
    cue_range = [2.2, 3.2]
    real_range = [1.1, 2.1]
    infer_range = [0, 1]
    color_list = [(0, 0, 0, alpha), (1, 0, 0, alpha),
                  (0, 0, 1, alpha), (0, 1, 0, alpha)]
    con_strings = sorted([column for column in pandata.columns
                          if (column.startswith('con')
                              and not column.startswith('con_'))])
    all_cons = np.concatenate([pandata['ix_context'].unique(),
                               np.arange(len(con_strings))])
    all_cons = np.unique(all_cons)
    colors = {idx: color
              for idx, color in zip(all_cons, color_list)}
    real_con = np.array(pandata['ix_context'])
    con_breaks = np.nonzero(np.diff(real_con))[0] + 1
    con_breaks = np.concatenate([[0], con_breaks, [len(real_con) - 1]])
    cons = np.array([real_con[one_break] for one_break in con_breaks])
    # plot real context
    for c_con, n_con, ix_con in zip(con_breaks, con_breaks[1:], cons):
        axis.fill_between([c_con, n_con], [real_range[0]] * 2,
                          [real_range[1]] * 2,
                          color=colors[ix_con])
    axis.text(x=len(pandata) / 2, y=1.6, s='Real context',
              horizontalalignment='center', verticalalignment='center')
    # ['con{}'.format(idx) for idx in all_cons]
    conx = np.array(pandata.loc[:, con_strings])
    conx = conx * (infer_range[1] - infer_range[0]) + infer_range[0]
    con_breaks = np.nonzero(np.diff(real_con))[0] + 1
    con_breaks = np.concatenate([[0], con_breaks, [len(real_con) - 1]])
    cons = np.array([real_con[one_break] for one_break in con_breaks])

    # plot cues
    real_cues = np.array(pandata['cue'])
    cue_breaks = np.nonzero(np.diff(real_cues))[0] + 1
    cue_breaks = np.concatenate([[0], cue_breaks, [len(real_cues) - 1]])
    cues = np.array([real_cues[one_break] for one_break in cue_breaks])
    for c_cue, n_cue, ix_cue in zip(cue_breaks, cue_breaks[1:], cues):
        axis.fill_between([c_cue, n_cue], *cue_range, color=colors[ix_cue])
    axis.text(x=len(pandata) / 2, y=np.mean(cue_range), s='Cues',
              horizontalalignment='center', verticalalignment='center')
    # plot inferred context
    for ix_con in range(len(con_strings)):
        color = colors[ix_con][:-1] + (1, )
        axis.plot(conx[:, ix_con], color=color)
    axis.text(x=len(pandata) / 2, y=0.5, s='Inferred context',
              horizontalalignment='center', verticalalignment='center')
    # Plot breaks
    for c_break in con_breaks:
        axis.plot([c_break, c_break], [infer_range[0], cue_range[1]],
                  color='black', alpha=0.2)
    if flag_makepretty:
        axis.set_xticks(con_breaks)
        axis.set_yticks([0, 0.5, 1])
        axis.set_title('Context inference')
        axis.set_xlabel('Trial')
        axis.set_ylabel('Prob. of context')
    plt.draw()


def plot_adaptation(pandata, axis=None, fignum=5):
    """Plots inferred magnitudes, hand position and "adaptation".

    Parameters
    ----------
    pandata : DataFrame
    Data from a simulation that contains the following columns:
      'pos(t)' : hand position
      'mag_mu_x' : Estimate of the magnitude of the force in context x,
                   for x = {0, 1, 2}.

    """
    columns = sorted(list(pandata.columns))
    trial = np.arange(len(pandata))
    colors = ['black', 'red', 'blue']
    adapt_color = np.array([174, 99, 164]) / 256
    if axis is None:
        fig, axis = plt.subplots(1, 1, clear=True, num=fignum)
    axis.plot(np.array(pandata['pos(t)']), color='black', alpha=0.4,
              label='Error')
    axis.plot(np.array(pandata['pos(t)'] + pandata['action']),
              color=adapt_color, label='Adaptation')
    magmu = [np.array(pandata[column])
             for column in columns
             if column.startswith('mag_mu')]
    errors = [np.array(pandata[column])
              for column in columns
              if column.startswith('mag_sd')]
    for color_x, error_x, magmu_x in zip(colors, errors, magmu):
        axis.plot(magmu_x, color=color_x, label='{} model'.format(color_x))
        axis.fill_between(trial, magmu_x - 2 * error_x, magmu_x + 2 * error_x,
                          color=color_x, alpha=0.1)
    magmu = np.array(magmu)
    yrange = np.array([magmu.min(), magmu.max()]) * 1.1
    axis.set_ylim(yrange)
    plt.draw()


def sim_and_plot(agent, pars_task, return_data=False,
                 force_labels=None, axes=None, fignum=6):
    """Simulates the agent playing and makes a nice plot with context inference and
    adaptation and colors everywhere.

    """
    pandata, pandagent, agent = thh.run(agent, pars=pars_task)
    pandota = thh.join_pandas(pandata, pandagent)
    if axes is None:
        fig, axes = plt.subplots(2, 1, num=fignum, clear=True, sharex=True)
    plot_adaptation(pandota, axis=axes[0])
    plot_contexts(pandota, axis=axes[1])

    fake_forces = np.array([-pars_task['forces'][1][1], 0, pars_task['forces'][1][1]])

    axes[0].set_ylabel('Adaptation (N)')
    # axes[0].set_yticks(ticks=fake_forces * 1.2)
    if force_labels is not None:
        axes[0].set_yticklabels(force_labels)
    axes[0].legend()

    axes[1].set_xlabel('Trial')
    axes[1].set_ylabel('p(context)', y=0.05, horizontalalignment='left')
    axes[1].set_yticks([0, 0.5, 1])

    if return_data:
        return pandota, agent


def slow_context_inference_nocues(prediction_noise=None, fignum=7):
    """Figure that shows the effects of slower and slower context inference in the
    absence of any cues, by virtue of noisy predictions from each model.

    """
    if prediction_noise is None:
        prediction_noise = 3
    pars_task = pars.task_oaoa
    agent = model.LRMeanSD(angles=[0, 0],
                           prediction_noise=prediction_noise,
                           cue_noise=0.5, context_noise=0,
                           force_sds=0.02 * np.ones(2))
    sim_and_plot(agent, pars_task, fignum=fignum)


def slow_context_inference_badcues(cue_noise=None, fignum=8):
    """Slow context inference based on bad cues and the interplay between cue noise
    and prediction noise.

    """
    if cue_noise is None:
        cue_noise = 0.017
    pars_task = pars.task_oa
    agent = model.LRMeanSD(angles=[0, 0], prediction_noise=0.0,
                           cue_noise=cue_noise, context_noise=0.1,
                           force_sds=0.01 * np.ones(2), max_force=5)
    sim_and_plot(agent, pars_task, fignum=fignum)


def baseline_bias(fignum=9):
    """This reproduces the experiments in Davidson_Scaling_2004, plotting the
    results for all their groups and experiments in one big pot of lovin.

    Note: for now, this does not work. The model would need some additions to
    handle the creation of new models and it goes beyond the scope of the
    paper.

    """
    fig, axes = plt.subplots(2, 4, gridspec_kw={'height_ratios': (3, 1)},
                             num=fignum, clear=True)
    agent = model.LRMeanSD(angles=[0, 0, 0],
                           prediction_noise=0,
                           cue_noise=1 / 3, context_noise=0.0,
                           force_sds=0.02 * np.ones(3),
                           max_force=1)
    pars_task = pars.task_davidson_1_1
    sim_and_plot(agent, pars_task, axes=[axes[0, 0], axes[1, 0]])

    agent = model.LRMeanSD(angles=[0, 0, 0],
                           prediction_noise=0,
                           cue_noise=1 / 3, context_noise=0.0,
                           force_sds=0.02 * np.ones(3),
                           max_force=1)
    pars_task = pars.task_davidson_1_2
    sim_and_plot(agent, pars_task, axes=[axes[0, 1], axes[1, 1]])

    agent = model.LRMeanSD(angles=[0, 0, 0],
                           prediction_noise=0,
                           cue_noise=1 / 3, context_noise=0.0,
                           force_sds=0.02 * np.ones(3),
                           max_force=1, hyper_sd=0.005)
    agent.all_learn = True
    pars_task = pars.task_davidson_2_1
    sim_and_plot(agent, pars_task, axes=[axes[0, 2], axes[1, 2]])

    agent = model.LRMeanSD(angles=[0, 0, 0],
                           prediction_noise=0,
                           cue_noise=1 / 3, context_noise=0.0,
                           force_sds=0.02 * np.ones(3),
                           max_force=1)
    pars_task = pars.task_davidson_2_2
    sim_and_plot(agent, pars_task, axes=[axes[0, 3], axes[1, 3]])
    axes[0, 0].get_legend().remove()
    axes[0, 1].get_legend().remove()
    axes[0, 2].get_legend().remove()
    axes[0, 3].get_legend().remove()


def oh_2019(plot=True, axes=None, fignum=10):
    """Simulates the experiments from Oh and Schweighoffer 2019.

    Parameters
    ----------
    plot : bool
    Whether to simulate and plot. If True, will create the agent
    with the parameters agent_pars and simulate with sim_and_plot().

    Returns
    -------
    task_20 : dictionary
    Parameters for the task with adaptation=20. To be used directly
    as the input for tth.run().

    task_10 : dictionary
    Same as --task_20-- but for adaptation=10.

    agent_pars : dictionary
    Parameters for the agent for both tasks. To be used directly with
    agent.RLMeanSD(). Note that it will not work for the other agents.

    """
    task_20 = task_pars.copy()
    task_20['obs_noise'] = 1.5
    task_20['force_noise'] = 1 * np.ones(2)
    task_20['forces'] = [[0, 0], [1, 20]]
    contexts_20 = [[0, 20], [1, 60], [0, 40], [1, 50], [0, 20],
                   [1, 30], [0, 40], [1, 50], [pars.CLAMP_INDEX, 120]]
    task_20['context_seq'] = pars.define_contexts(contexts_20)
    task_20['breaks'] = np.zeros(len(task_20['context_seq']))
    task_20['cues'] = np.zeros(len(task_20['context_seq']), dtype=int)

    agent_pars = {'angles': [0, 0],
                  'cue_noise': 1 / 2,
                  'max_force': 50,
                  'hyper_sd': 1,
                  'obs_sd': 2.5,
                  'context_noise': 0.05,
                  'force_sds': np.ones(2),
                  'prediction_noise': 1}

    task_10 = task_20.copy()
    task_10['forces'] = [[0, 0], [1, 10]]
    if plot:
        if axes is None:
            fig, axes = plt.subplots(2, 2, num=fignum, clear=True,
                                     sharex=True, sharey=False)

        agent = model.LRMeanSD(**agent_pars)

        agent.all_learn = True
        sim_and_plot(agent, task_20, axes=axes[:, 0],
                     force_labels=[-20, 0, 20])
        axes[0, 0].set_title('Adaptation: 20')
        agent = model.LRMeanSD(**agent_pars)
        agent.all_learn = True
        sim_and_plot(agent, task_10, axes=axes[:, 1],
                     force_labels=[-10, 0, 10])
        axes[0, 1].set_title('Adaptation: 10')

    return task_20, task_10, agent_pars


def kim_2015(plot=True, axis=None, fignum=11):
    """Simulates an experiment from Kim et al. 2015.

    Parameters
    ----------
    plot : bool
    Whether to simulate and plot. If True, will create the agent
    with the parameters agent_pars and simulate with sim_and_plot().

    Returns
    -------
    task : dictionary
    Parameters for the task. To be used directly as the input for tth.run().

    agent_pars : dictionary
    Parameters for the agent for both tasks. To be used directly with
    agent.RLMeanSD(). Note that it will not work for the other agents.

    """
    task = task_pars.copy()
    task['obs_noise'] = 3
    task['force_noise'] = 0.01 * np.ones(3)
    task['forces'] = [[0, 0], [-1, 40], [1, 40]]
    blocks = [0, 1, 2, 1, 2, 0, 2, 1, 2, 1, 0, 0, 2, 1, 2, 1,
              0, 1, 2, 1, 2, 0, 0, 1, 2, 1, 2, 0, 2, 1, 2, 1, 0]
    contexts = [[idx, 30] for idx in blocks]
    task['context_seq'] = pars.define_contexts(contexts)
    task['breaks'] = np.zeros(len(task['context_seq']))
    task['cues'] = task['context_seq']

    agent_pars = {'angles': [0, -1, 1],
                  'prediction_noise': 0,
                  'cue_noise': 0.0001,
                  'context_noise': 0.1,
                  'force_sds': np.ones(3),
                  'max_force': 60,
                  'hyper_sd': 1,
                  'obs_sd': 3}
    if plot:
        agent = model.LRMeanSD(**agent_pars)
        sim_and_plot(agent, task, force_labels=[-40, 0, 40],
                     fignum=fignum)
    return task, agent_pars


def herzfeld_2014(plot=True, axes=None, fignum=12):
    """Simulates something like the experiments from Herzfeld et al. 2014.

    This does two conditions: high uncertainty and low uncertainty, where
    the uncertainty is over the current context given the previous one. In
    the high uncertainty condition, there's a 50/50 chance of a context
    change. In the low uncertainty, there is a 0.1 chance of change.

    """
    num_trials = 300
    task = task_pars.copy()
    task['obs_noise'] = 2
    task['force_noise'] = 0.01 * np.ones(3)
    task['forces'] = [[0, 0], [-1, 13], [1, 13]]
    task['breaks'] = np.zeros(num_trials, dtype=int)
    task['cues'] = np.zeros(num_trials, dtype=int)

    task_high = task.copy()
    contexts = np.random.choice([1, 2], size=num_trials)
    task_high['context_seq'] = contexts
    task_low = task.copy()
    switches = np.random.choice([0, 1], p=(0.99, 0.01), size=num_trials)
    switches_ct = np.concatenate([[0], np.nonzero(switches)[0], [num_trials]])
    switches_delta = np.diff(switches_ct)
    contexts = [[(idx % 2) + 1, switch] for idx, switch in enumerate(switches_delta)]
    task_low['context_seq'] = pars.define_contexts(contexts)

    agent_pars_high = {'angles': [0, -5, 5],
                       'prediction_noise': 0.1,
                       'cue_noise': 1 / 3,
                       'context_noise': 0.25,
                       'force_sds': 0.1 * np.ones(3),
                       'max_force': 20,
                       'hyper_sd': 1000,
                       'obs_sd': 2}
    agent_pars_low = agent_pars_high.copy()
    agent_pars_low['context_noise'] = 0.1
    if plot:
        if axes is None:
            fig, axes = plt.subplots(2, 2, num=fignum, clear=True,
                                     sharex=True, sharey=False)

        agent = model.LRMeanSD(**agent_pars_high)
        agent.all_learn = True
        agent.threshold_learn = 0.2
        sim_and_plot(agent, task_high, axes=axes[:, 0])
        agent = model.LRMeanSD(**agent_pars_low)
        agent.all_learn = True
        agent.threshold_learn = 0.2
        sim_and_plot(agent, task_low, axes=axes[:, 1])
        
    return task_high, task_low, agent_pars_high, agent_pars_low


def davidson_2004(plot=True, axes=None, fignum=13):
    """Simulates the second experiment in Davidson_Scaling_2004."""
    task_m8 = task_pars.copy()
    task_m8['obs_noise'] = 0.1
    task_m8['force_noise'] = 0.5 * np.ones(3)
    task_m8['forces'] = [[0, 0], [1, 4], [-1, 4]]
    task_m8['context_seq'] = pars.define_contexts([[0, 10],
                                                   [1, 100],
                                                   [2, 100]])
    task_m8['cues'] = np.zeros(len(task_m8['context_seq']), dtype=int)
    task_m8['breaks'] = np.zeros(len(task_m8['context_seq']), dtype=int)

    task_p8 = task_m8.copy()
    task_p8['forces'] = [[0, 0], [1, 4], [1, 12]]

    agent_pars = {'angles': [0, 5, 0],
                  'cue_noise': 1 / 3,
                  'max_force': 20,
                  'hyper_sd': 1,
                  'obs_sd': 2,
                  'context_noise': 0.05,
                  'force_sds': 0.5 * np.ones(3),
                  'prediction_noise': 0}
    if plot:
        if axes is None:
            fig, axes = plt.subplots(2, 2, num=fignum, clear=True,
                                     sharex=True, sharey=False)

        agent = model.LRMeanSD(**agent_pars)
        agent.all_learn = True
        agent.threshold_learn = 0.2
        sim_and_plot(agent, task_m8, axes=axes[:, 0])
        axes[0, 0].set_title('-8 group')
        axes[0, 0].set_ylim([-12, 12])
        agent = model.LRMeanSD(**agent_pars)
        agent.all_learn = True
        agent.threshold_learn = 0.2
        sim_and_plot(agent, task_p8, axes=axes[:, 1])
        axes[0, 1].set_title('+8 group')
        axes[0, 1].set_ylim([-12, 12])

    return task_m8, task_p8, agent_pars


if __name__ == '__main__':
    grid_sims()

# Huang_2011
# Zarahn_2008 (huge adaptation)
# Sing_2010 weighted actions anterograde inference
