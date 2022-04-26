# -*- coding: utf-8 -*-
# .adaptation/code/figures.py

"""Figures for the context inference motor adaptation paper. The default
parameter --fignum-- for each function defined here determines which figure it
is in the paper. Those that do not have one are not paper figures, but
auxiliary functions."""

import ipdb
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib import gridspec as gs
import seaborn as sns
import matplotlib as mpl

import simulations as sims
import model
import task_hold_hand as thh
import pars

FIGURE_FOLDER = '../article/figures/'

mpl.rcParams['font.size'] = 8
mpl.rcParams['figure.titlesize'] = 'small'
mpl.rcParams['lines.linewidth'] = 1


def model_showoff(fignum=1, show=True, do_a=True):
    """Plots a bunch of arbitrary simulationsthat show how the parameters of
    the model work and affect inference. Also leaves an empty space on top
    to insert the diagram of the generative model by hand.

    Parameters
    ----------
    do_a : bool
    Whether to add the model diagram to the figure. If True, the model diagram
    is assumed to be in FIGURE_FOLDER/generative.png. If False, the space will
    be left empty at the top.

    """
    runs = 50
    colors = ['black', 'tab:green', 'tab:blue']
    cycler = plt.cycler('color', colors)
    tasks, agents_pars = sims.model_showoff(plot=False)
    numbers = tasks.shape
    figsize = (6, 8)
    fig = plt.figure(fignum, clear=True, figsize=figsize)
    height_ratios = np.ones(numbers[0] + 2)
    height_ratios[0] = 3
    height_ratios[1] = 0.3
    gsbig = gs.GridSpec(numbers[0] + 2, numbers[1], figure=fig,
                        height_ratios=height_ratios, hspace=0.1, wspace=0.1)
    axis_diagram = fig.add_subplot(gsbig[0, :])
    ctx_switch = np.diff(tasks.reshape(-1)[0]['context_seq']).nonzero()[0][0] + 1
    num_trials = len(tasks.reshape(-1)[0]['context_seq'])
    axes = np.empty((*numbers, 2), dtype=object)
    small_hr = [2, 1]  # height ratios for adaptation/p(ctx)
    for ix_row in np.arange(2, numbers[0] + 2):
        for ix_col in np.arange(numbers[1]):
            gssmall = gsbig[ix_row, ix_col].subgridspec(2, 1, hspace=0.05,
                                                        height_ratios=small_hr)
            axes[ix_row - 2, ix_col, 0] = fig.add_subplot(gssmall[0])
            axes[ix_row - 2, ix_col, 1] = fig.add_subplot(gssmall[1])
    for ix_row in range(numbers[0]):
        for ix_col in range(numbers[1]):
            c_axes = axes[ix_row, ix_col, :]
            agent = model.LRMeanSD(**agents_pars[ix_row, ix_col])
            task = tasks[ix_row, ix_col]
            pandata, pandagent, _ = thh.run(agent, pars=task)
            pandota = thh.join_pandas(pandata, pandagent)
            pandota = sims.multiple_runs(runs=runs,
                                         agent_pars=[agents_pars[ix_row,
                                                                 ix_col], ],
                                         task_pars=[task, ])
            c_axes[0].axvline(ctx_switch, linestyle='--', color='black',
                              alpha=0.3)
            c_axes[1].axvline(ctx_switch, linestyle='--', color='black',
                              alpha=0.3)
            c_axes[1].set_prop_cycle(cycler)
            sns.lineplot(data=pandota, x='trial', y='con0', ci='sd',
                         ax=c_axes[1])
            sns.lineplot(data=pandota, x='trial', y='con1', ci='sd',
                         ax=c_axes[1])
            plot_adaptation(pandota, axis=c_axes[0], colors=colors)
            # c_axes[0].set_ylim((-3, 6))
    yticks_ada = axes[-1, 0, 0].get_yticks()[1:]
    xticks = np.array([0, ctx_switch])
    for axis in axes.reshape(-1):
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlim((0, num_trials - 1))
        axis.set_xlabel('')
        axis.set_ylabel('')
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
    for rows in axes[-1, :, :]:
        rows[1].set_xlabel('Trial')
        rows[1].set_xticks(xticks)
        # rows[1].set_xlim((0, num_trials))
    for cols in axes[:, 0, :]:
        cols[0].set_yticks(yticks_ada)
        cols[1].set_yticks([0, 0.5], labels=['0', '.5'])
    cue_texts = ['Low', 'Cue uncertainty\nMed', 'High']
    for tops, text in zip(axes[0, :, 0], cue_texts):
        tops.set_title(text, verticalalignment='bottom')
    centering = (1 - small_hr[1] / small_hr[0]) / 2
    obs_texts = ['Low', 'Med.', 'High']
    for lefts, text in zip(axes[:, -1, 0], obs_texts):
        lefts.text(s=text, x=1.1, y=centering, transform=lefts.transAxes,
                   rotation=270, horizontalalignment='center',
                   verticalalignment='center')
    axes[1, -1, 0].text(s='Obs. Noise', x=1.2, y=centering,
                        transform=axes[1, -1, 0].transAxes,
                        rotation=270, verticalalignment='center')
    for axis in axes[:, :, 0].reshape(-1):
        axis.set_ylim((-1, 5))
        axis.axline((0, 4), slope=0, color='black', linestyle='--', alpha=0.3)
    axes[-1, 0, 1].set_ylabel(r'p(ctx)')
    axes[-1, 0, 0].set_ylabel(r'Adaptation')

    # Import diagram of the model from png
    if do_a:
        diagram = plt.imread(FIGURE_FOLDER + '/generative.png')
        axis_diagram.axis('off')
        axis_diagram.imshow(diagram)

    # subplot labels
    offset_multiplier = 0.03
    offset = np.array([-1, figsize[1] / figsize[0]]) * offset_multiplier
    anchor_a = np.array(axis_diagram.get_position())[[0, 1], [0, 1]] + offset
    anchor_b = np.array(axes[0, 0, 0].get_position())[[0, 1], [0, 1]] + offset
    fig.text(s='A', x=anchor_a[0], y=anchor_a[1], fontdict={'size': 12})
    fig.text(s='B', x=anchor_b[0], y=anchor_b[1], fontdict={'size': 12})

    plt.savefig(FIGURE_FOLDER + 'figure_{}.png'.format(fignum), dpi=600)
    plt.savefig(FIGURE_FOLDER + 'figure_{}.svg'.format(fignum), format='svg')
    if show:
        plt.draw()
        plt.show(block=False)


def plot_adaptation(pandata, axis, colors):
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
    adapt_color = np.array([174, 99, 164]) / 256
    # axis.plot(np.array(pandata['pos(t)']), color='black', alpha=0.4,
    #           label='Error')
    magmu = [np.array(pandata[column])
             for column in columns
             if column.startswith('mag_mu')]
    errors = [np.array(pandata[column])
              for column in columns
              if column.startswith('mag_sd')]
    for color_x, error_x, magmu_x in zip(colors, errors, magmu):
        axis.plot(magmu_x, color=color_x, label='{} model'.format(color_x),
                  linewidth=2)
        # axis.fill_between(trial, magmu_x - 2 * error_x, magmu_x + 2 * error_x,
        #                   color=color_x, alpha=0.1)
    axis.plot(np.array(-pandata['hand'] - pandata['action']),
              color=adapt_color, label='Adaptation', zorder=1000)
    magmu = np.array(magmu)
    yrange = np.array([magmu.min(), magmu.max()]) * 1.1
    axis.set_ylim(yrange)
    plt.draw()


def oh_2019_kim_2015(fignum=2, show=True):
    """Reproduces the results from Oh_Minimizing_2019 and kim_neural_2015 and
    plots them in a format similar to their plots.

    An empty subplot is created to manually add the figure from their
    paper.

    """
    context_color = np.ones(3) * 0.5
    ad_color = np.ones(3) * 0.3
    cycler = plt.cycler('color', ['black', 'tab:green', 'tab:blue'])

    figsize = (6, 4)
    fig = plt.figure(fignum, clear=True, figsize=figsize)
    magri = gs.GridSpec(3, 4, width_ratios=[1, 0.15, 1, 1], wspace=0.05,
                        hspace=0.25, figure=fig)
    axes = np.empty((3, 3), dtype=object)
    for ix_col, col in enumerate([0, 2, 3]):
        for ix_row in range(3):
            if ix_col == 2:
                sharex = axes[ix_row, ix_col - 1]
                sharey = axes[ix_row, ix_col - 1]
            else:
                sharex = sharey = None
            axes[ix_row, ix_col] = fig.add_subplot(magri[ix_row, col],
                                                   sharex=sharex,
                                                   sharey=sharey)

    # a)
    trials_kim = np.arange(300)  # It IS used in the query below
    task_kim, agent_pars = sims.kim_2015(plot=False)
    agent_kim = model.LRMeanSD(**agent_pars)
    pandata, pandagent, _ = thh.run(agent_kim, pars=task_kim)
    pandota_kim = thh.join_pandas(pandata, pandagent)
    pandota_kim = pandota_kim.query('trial in @trials_kim')
    kimcon = ['con0', 'con1', 'con2']
    contexts = pandota_kim['ix_context'].values
    contexts[contexts == pars.CLAMP_INDEX] = 0
    contexts[contexts == 1] = -40
    contexts[contexts == 2] = 40
    axes[0, 1].plot(contexts, color=context_color)
    axes[0, 2].plot(contexts, color=context_color)
    axes[0, 1].plot(-pandota_kim['pos(t)'] - pandota_kim['action'],
                    color=ad_color)
    pcons = np.array(pandota_kim.loc[:, kimcon])
    axes[0, 0].set_prop_cycle(cycler)
    axes[0, 0].plot(pcons)
    axes[0, 1].set_yticks([-40, 0, 40])
    axes[0, 1].text(s='Angle', x=-0.2, y=0.5,
                    transform=axes[0, 1].transAxes, rotation=90,
                    horizontalalignment='center',
                    verticalalignment='center')
    axes[0, 1].set_title('Adaptation')
    axes[0, 0].set_title('p(context)')

    # b)
    ohcon = ['con0', 'con1']
    task_20, task_10, agent_pars = sims.oh_2019(plot=False)
    agent = model.LRMeanSD(**agent_pars)
    agent.all_learn = True
    pandata, pandagent, _ = thh.run(agent, pars=task_20)
    pandota_20 = thh.join_pandas(pandata, pandagent)
    contexts = pandota_20['ix_context'].values
    contexts[contexts == pars.CLAMP_INDEX] = 0
    contexts[contexts == 1] = 20
    contexts_20 = contexts
    switches = np.nonzero(np.diff(contexts_20))[0]
    switches = np.concatenate([[0], switches, [len(contexts)]])
    axes[1, 1].plot(contexts, color=context_color)
    axes[1, 2].plot(contexts, color=context_color)
    axes[1, 1].plot(-pandota_20['pos(t)'] - pandota_20['action'],
                    color=ad_color)
    axes[1, 0].set_prop_cycle(cycler)
    axes[1, 0].plot(pandota_20.loc[:, ohcon])
    axes[1, 1].text(s='Angle', x=-0.2, y=0.5,
                    transform=axes[1, 1].transAxes, rotation=90,
                    horizontalalignment='center',
                    verticalalignment='center')
    axes[1, 0].set_ylim([0, 1.2])
    ylim = axes[1, 0].get_ylim()
    axes[1, 0].vlines(switches, *ylim, linestyle='--', color='black',
                      linewidth=0.5)

    # c)
    agent = model.LRMeanSD(**agent_pars)
    agent.all_learn = True
    pandata, pandagent, _ = thh.run(agent, pars=task_10)
    pandota_10 = thh.join_pandas(pandata, pandagent)
    contexts = pandota_10['ix_context'].values
    contexts[contexts == pars.CLAMP_INDEX] = 0
    contexts[contexts == 1] = 10
    contexts_10 = contexts
    axes[2, 1].plot(contexts_10, color=context_color)
    axes[2, 2].plot(contexts_10, color=context_color)
    axes[2, 1].plot(-pandota_10['pos(t)'] - pandota_10['action'],
                    color=ad_color)
    axes[2, 1].set_xlabel('Trials')
    axes[2, 1].text(s='Angle', x=-0.2, y=0.5,
                    transform=axes[2, 1].transAxes, rotation=90,
                    horizontalalignment='center',
                    verticalalignment='center')
    axes[2, 0].set_prop_cycle(cycler)
    axes[2, 0].plot(pandota_10.loc[:, ohcon])
    axes[2, 0].set_xlabel('Trial')
    axes[2, 0].set_ylim([0, 1.2])
    ylim = axes[2, 1].get_ylim()
    axes[2, 0].vlines(switches, *ylim, linestyle='--', color='black',
                      linewidth=0.5)

    # Subplot labels
    axes[0, 0].text(x=-0.1, y=1.05, s='A',
                    transform=axes[0, 0].transAxes,
                    fontdict={'size': 12})
    axes[1, 0].text(x=-0.1, y=1.05, s='B',
                    transform=axes[1, 0].transAxes,
                    fontdict={'size': 12})
    axes[2, 0].text(x=-0.1, y=1.05, s='C',
                    transform=axes[2, 0].transAxes,
                    fontdict={'size': 12})

    # Axes shenanigans
    for axis in axes[:, 0].reshape(-1):
        axis.set_yticks([0, 1])
    for axis in axes[:, -1]:
        yticks = axis.get_yticks().astype(int)
        axis.set_yticklabels(yticks, visible=False)
    axes[0, -1].set_title('Exp. data')
    axes[-1, -1].set_xlabel('Trial')
    for row in [0, 1]:
        for axis in axes[row, :].reshape(-1):
            axis.set_xticklabels(axis.get_xticklabels(), visible=False)
    for axis in axes.reshape(-1):
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
    # fig.align_ylabels(axes[:, 0])
    plt.savefig(FIGURE_FOLDER + 'figure_{}.png'.format(fignum), dpi=600)
    plt.savefig(FIGURE_FOLDER + 'figure_{}.svg'.format(fignum), format='svg')
    if show:
        plt.draw()
        plt.show(block=False)


def davidson_2004(fignum=3, show=True):
    """Reproduces the results from Davidson_Scaling_2004, leaving an empty
    subplot to put in the results from their paper.

    """
    repeats = 8  # No. of participants per group
    figsize = (6, 4)
    colors = [np.array((95, 109, 212)) / 256,
              np.array((212, 198, 95)) / 256]
    # colors = {'-A': np.array((95, 109, 212)) / 256,
    #           '3A': np.array((212, 198, 95)) / 256}
    ran = [161, 201]  # Trials of importance for this plot

    fig = plt.figure(num=fignum, clear=True, figsize=figsize)
    gsbig = gs.GridSpec(2, 1, figure=fig, height_ratios=[1.5, 1], hspace=0.4)
    gstop = gsbig[0].subgridspec(2, 4, width_ratios=[1, 0.25, 2, 2],
                                 wspace=0.05)
    gsbot = gsbig[1].subgridspec(2, 4, width_ratios=[1, 1.8, 1, 1.8],
                                 wspace=0.25)

    axes = []  # Gets turned into np.array later
    axes.append(fig.add_subplot(gstop[:, 2]))
    axes.append(fig.add_subplot(gstop[0, 0]))
    axes.append(fig.add_subplot(gstop[1, 0]))
    axes.append(fig.add_subplot(gstop[:, 3]))
    axes = np.array(axes)

    baxes = []  # Gets turned into np.array later
    baxes.append(fig.add_subplot(gsbot[:, 1]))
    baxes.append(fig.add_subplot(gsbot[0, 0]))
    baxes.append(fig.add_subplot(gsbot[1, 0]))
    baxes.append(fig.add_subplot(gsbot[:, 3]))
    baxes.append(fig.add_subplot(gsbot[0, 2]))
    baxes.append(fig.add_subplot(gsbot[1, 2]))
    baxes = np.array(baxes)

    tasks_all, agents_pars_all = sims.davidson_2004(plot=False)
    agent_pars = agents_pars_all[:2]  # For top row
    tasks = tasks_all[:2]             # For top row
    agent_sims = agents_pars_all[2:]  # For bottom row
    tasks_sims = tasks_all[2:]        # For bottom row

    names = ['-A', '3A']
    labels = [['O', '-A', 'A'], ['O', '3A', 'A']]
    _davidson_trio(repeats, colors, agent_pars, tasks, names, axes, labels,
                   ran)
    names = ['-2A', '4A']
    labels = [['_O', '-2A', '_A'], ['_O', '4A', '_A']]
    _davidson_trio(repeats, colors, agent_sims[:2], tasks_sims[:2],
                   names, baxes[:3], labels, ran)
    names = ['-A', '3A']
    labels = [['_O', '-A', 'A'], ['_O', '3A', 'A']]
    _davidson_trio(repeats, colors, agent_sims[2:], tasks_sims[2:],
                   names, baxes[3:], labels, ran)
    for axis in axes.reshape(-1):
        axis.set_xlabel('')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    axes[3].set_xlabel('Trials since switch')
    for axis in baxes.reshape(-1):
        axis.set_xlabel('')
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
    axes[1].set_title('p(ctx)', horizontalalignment='center')
    axes[3].set_title('Exp. data')
    axes[3].set_yticklabels([])
    baxes[0].set_xlabel('Trials since switch')
    baxes[4].set_ylabel('')
    baxes[5].set_ylabel('')
    baxes[1].set_xticks([])
    baxes[4].set_xticks([])

    # subplot labels
    offset_multiplier = 0.03
    offset = np.array([-1, figsize[1] / figsize[0]]) * offset_multiplier
    anchor_a = np.array(axes[1].get_position())[[0, 1], [0, 1]] + offset
    anchor_b = np.array(baxes[1].get_position())[[0, 1], [0, 1]] + offset
    anchor_c = np.array(baxes[4].get_position())[[0, 1], [0, 1]] + offset
    fig.text(s='A', x=anchor_a[0], y=anchor_a[1], fontdict={'size': 12})
    fig.text(s='B', x=anchor_b[0], y=anchor_b[1], fontdict={'size': 12})
    fig.text(s='C', x=anchor_c[0], y=anchor_c[1], fontdict={'size': 12})

    plt.savefig(FIGURE_FOLDER + 'figure_{}.png'.format(fignum), dpi=600)
    plt.savefig(FIGURE_FOLDER + 'figure_{}.svg'.format(fignum), format='svg')
    if show:
        plt.draw()
        plt.show(block=False)


def _davidson_trio(repeats, color_list, agent_pars, tasks, names, axes, labels,
                   ran):
    """Top row of plots for davidson_2004().

    There should be three axes. The last two are used to plot context
    inference, the first for adaptation. --agent_pars--, --names-- and
    --tasks-- are assumed to have the same size (2, ), corresponding to the two
    groups in each simulated experiment.

    --labels-- contains the labes for each plot, which means that there have to
      be three labels per each agent, e.g. [['a', 'b', 'c'], ...]. Note that
      prefixing a label with an underscore makes it disappear from the legend.

    """
    colors = {key: color for key, color in zip(names, color_list)}
    data = {name: [] for name in names}
    for idx, (task, ag_pars, name) in enumerate(zip(tasks, agent_pars, names)):
        for idx in range(repeats):
            agent = model.LRMeanSD(**ag_pars)
            pandata, pandagent, _ = thh.run(agent, pars=task)
            pandota = thh.join_pandas(pandata, pandagent)
            pandota['part'] = idx
            data[name].append(pandota)
        data[name] = pd.concat(data[name])
        data[name]['pos(t)'] = np.abs(data[name]['pos(t)'])
    data = pd.concat(data, axis=0, names=['Group', 'trial'])
    data.reset_index('Group', inplace=True)
    data.reset_index('trial', inplace=True)
    ymax = data.loc[data['trial'] > ran[0], 'pos(t)'].max()
    ymin = data.loc[data['trial'] > ran[0], 'pos(t)'].min()
    sns.lineplot(data=data, x='trial', y='pos(t)', ax=axes[0],
                 hue='Group', palette=colors, ci='sd')
    datum_a = data.query('Group == @names[0]')
    datum_b = data.query('Group == @names[1]')
    sns.lineplot(data=datum_a, x='trial', y='con0', ax=axes[1],
                 color='black', label=labels[0][0])
    sns.lineplot(data=datum_a,
                 x='trial', y='con1', ax=axes[1], label=labels[0][1],
                 color='tab:green')
    sns.lineplot(data=datum_a,
                 x='trial', y='con2', ax=axes[1], label=labels[0][2],
                 color='tab:blue')
    sns.lineplot(data=datum_b, x='trial', y='con0', ax=axes[2],
                 color='black', label=labels[1][0])
    sns.lineplot(data=datum_b,
                 x='trial', y='con1', ax=axes[2], label=labels[1][1],
                 color='tab:green')
    sns.lineplot(data=datum_b,
                 x='trial', y='con2', ax=axes[2], label=labels[1][2],
                 color='tab:blue')
    axes[1].legend(ncol=1, fontsize='x-small', handlelength=1)
    axes[2].legend(ncol=1, fontsize='x-small', handlelength=1)

    ticks = np.array(axes[0].get_xticks(), dtype=int) - ran[0] + 1
    for idx in range(len(axes)):
        axes[idx].set_xlim(ran)
        ticks = np.array(axes[idx].get_xticks(), dtype=int) - ran[0] + 1
        axes[idx].set_xticklabels(ticks)
    axes[1].set_ylabel('Grp. -A', labelpad=0)
    axes[2].set_ylabel('Grp. 3A', labelpad=0)
    axes[0].set_ylabel('Error (a.u.)', labelpad=0)
    axes[0].set_yticklabels([])
    axes[1].set_yticks((0, 1))
    axes[2].set_yticks((0, 1))
    axes[0].set_ylim(1.2 * np.array((ymin, ymax)))


def vaswani_2013(fignum=4, show=True, pandota=None):
    """Reproduces the results from Vaswani_Decay_2013, specifically their
    figures 2a-c.

    """
    reps = 10
    figsize = (6, 6)
    colors = ['blue', 'green', 'red', 'c']
    context_colors = ['black', 'tab:orange', 'tab:purple', 'tab:brown',
                      'tab:olive']
    mags = gs.GridSpec(4, 4, height_ratios=[2.2, 0.6, 1, 1],
                       width_ratios=[1, 1, 0.5, 2.2], wspace=0.05, hspace=0.05)
    fig = plt.figure(num=fignum, clear=True, figsize=figsize)
    axes_con = [fig.add_subplot(mags[2, 0])]
    axes_con.append(fig.add_subplot(mags[2, 1]))
    axes_con.append(fig.add_subplot(mags[3, 0]))
    axes_con.append(fig.add_subplot(mags[3, 1]))
    axes_sum = [fig.add_subplot(mags[0, 0:2]), ]
    axes_sum.append(fig.add_subplot(mags[0, 3], sharex=axes_sum[0]))
    axes_lag = fig.add_subplot(mags[2:4, 3])
    tasks, agents = sims.vaswani_2013(plot=False)

    all_pandas = []
    names = [1.1, 1.2, 1.3, 1.4]
    named_colors = {name: color for name, color in zip(names, colors)}
    for idx, (task, agent_par) in enumerate(zip(tasks, agents)):
        for ix_rep in range(reps):
            agent = model.LRMeanSD(**agent_par)
            pandata, pandagent, _ = thh.run(agent=agent,
                                            pars=task)
            c_pandota = thh.join_pandas(pandata, pandagent)
            c_pandota['group'] = names[idx]
            c_pandota['run'] = idx * reps + ix_rep  # to separate all runs
            c_pandota.reset_index('trial', inplace=True)
            all_pandas.append(c_pandota)
    pandota = pd.concat(all_pandas, axis=0, ignore_index=True)
    adaptation = -pandota['pos(t)'] - pandota['action']
    pandota['adaptation'] = adaptation
    pandota.loc[pandota['group'] == 'Group 1.4', ['adaptation']] *= -1
    # Plot context inference
    for idx, (group, color, ax) in enumerate(zip(names, colors, axes_con)):
        c_panda = pandota.query('group == @group')
        real_con = np.array(c_panda.groupby('trial').mean()['ix_context'])
        con_breaks = np.nonzero(np.diff(real_con))[0] + 1
        con_breaks = np.concatenate([[0], con_breaks, [len(real_con) - 1]])
        cons = np.array([real_con[one_break] for one_break in con_breaks],
                        dtype=int)
        # plot real context
        for c_con, n_con, ix_con in zip(con_breaks, con_breaks[1:], cons):
            if ix_con == pars.CLAMP_INDEX:
                ix_con = -1
            ax.fill_between([c_con, n_con], [0] * 2,
                            [1] * 2,
                            color=context_colors[ix_con],
                            alpha=0.2)
        con_strings = sorted([column for column in c_panda.columns
                              if (column.startswith('con')
                                  and not column.startswith('con_'))])
        for ix_con, con in enumerate(con_strings):
            # Little hack to get the colors to match between real and inferred:
            c_con = ix_con
            if idx == 2 and ix_con == 2:
                c_con = 3
            if idx == 3 and ix_con == 1:
                c_con = 2
            # End little hack.
            sns.lineplot(data=c_panda, x='trial', y=con,
                         color=context_colors[c_con],
                         ax=ax, ci='sd')
    for idx in range(4):
        axes_con[idx].text(s='Group {}'.format(names[idx]),
                           x=0.5, y=0.95, horizontalalignment='center',
                           verticalalignment='top',
                           transform=axes_con[idx].transAxes)
        axes_con[idx].set_xlabel('')
        axes_con[idx].set_ylabel('')
        axes_con[idx].set_xticks([])
        axes_con[idx].set_yticks([])
        axes_con[idx].set_ylim([0, 1])
    axes_con[2].set_yticks([0, 1])
    axes_con[2].set_ylabel(r'p(ctx)', labelpad=-0.1)
    axes_con[2].set_xticks([75, 175], [0, 100])
    axes_con[2].set_xlabel('Trial')
    axes_con[0].text(x=-0.3, y=1.1, s='C',
                     transform=axes_con[0].transAxes,
                     fontdict={'size': 12})

    # Plot summary adaptation:
    condi = 'trial >= 75 and trial <= 175 and group != 1.4'
    pandota_e = pandota.query(condi)
    # pandota_e.reset_index('trial', inplace=True)
    pandota_e.loc[:, 'trial'] -= 100
    sns.lineplot(x='trial', y='adaptation', hue='group',
                 palette=named_colors, data=pandota_e,
                 ax=axes_sum[0], ci='sd')
    # labels = [name[-3:] for name in names[:-1]]
    # axes_sum[0].legend(ncol=3, labels=labels)
    y_range = axes_sum[0].get_ylim()
    axes_sum[0].plot([0, 0], y_range, linestyle='dashed',
                     color='black', alpha=0.3)
    axes_sum[1].plot([0, 0], y_range, linestyle='dashed',
                     color='black', alpha=0.3)
    axes_sum[0].set_ylim(y_range)
    axes_sum[1].set_ylim(y_range)
    axes_sum[0].set_yticks([0, 1])
    axes_sum[1].set_yticks([])
    axes_sum[0].set_xlabel('Trials since start of error-clamp', labelpad=1)
    axes_sum[1].set_xlabel('Trials since start of error-clamp', labelpad=1)
    axes_sum[0].set_ylabel('Adaptation index')

    axes_sum[0].text(x=-0.1, y=1, s='A',
                     transform=axes_sum[0].transAxes,
                     fontdict={'size': 12})
    axes_sum[1].text(x=-0.1, y=1, s='B',
                     transform=axes_sum[1].transAxes,
                     fontdict={'size': 12})

    # Plot lags
    panda_lag = pandota.query('trial > 90 and trial <= 140')
    # panda_lag.reset_index('trial', inplace=True)
    panda_lag.loc[:, 'trial'] -= 100
    axes_lag.set_ylabel('p(ctx)')
    axes_lag.text(x=-0.1, y=1.033, s='D',
                  transform=axes_lag.transAxes,
                  fontdict={'size': 12})

    sns.lineplot(data=panda_lag, x='trial', y='con1',
                 hue='group', units='run', estimator=None,
                 palette=colors)
    axes_lag.set_xlabel('Trials since start of error-clamp')

    axes_sum[0].set_title('Simulations')
    axes_sum[1].set_title('Exp. data')
    axes_lag.set_title('Simulations')
    
    plt.savefig(FIGURE_FOLDER + 'figure_{}.png'.format(fignum), dpi=600)
    plt.savefig(FIGURE_FOLDER + 'figure_{}.svg'.format(fignum), format='svg')

    if show:
        plt.draw()
        plt.show(block=False)


def test(fignum=100, figsize=(5, 5)):
    mags = gs.GridSpec(3, 3, height_ratios=[2.2, 1, 1],
                       width_ratios=[1, 1, 2.2], wspace=0.1, hspace=0.1)
    fig = plt.figure(num=fignum, clear=True, figsize=figsize)
    axes_con = [fig.add_subplot(mags[1, 0])]
    axes_con.append(fig.add_subplot(mags[1, 1], sharey=axes_con[0]))
    axes_con.append(fig.add_subplot(mags[2, 0], sharey=axes_con[0]))
    axes_con.append(fig.add_subplot(mags[2, 1], sharey=axes_con[0]))
    axes_sum = [fig.add_subplot(mags[0, 0:2]), ]
    axes_sum.append(fig.add_subplot(mags[0, 2], sharex=axes_sum[0]))
    axes_lag = fig.add_subplot(mags[1:3, 2])

    all_axes = [*axes_con, *axes_sum, axes_lag]
    for axis in all_axes:
        axis.plot([0, 1], [0, 1], color='blue')
        axis.set_ylabel('yo')
    plt.draw()
    plt.show(block=False)
