# -*- coding: utf-8 -*-
# .adaptation/code/figures.py

"""Figures for the context inference motor adaptation paper. The default
parameter --fignum-- for each function defined here determines which figure it
is in the paper. Those that do not have one are not paper figures, but
auxiliary functions."""


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib import gridspec as gs
import seaborn as sns

import simulations as sims
import model
import task_hold_hand as thh
import pars

FIGURE_FOLDER = '../article/figures/'


# Note: Figure 1 is made by hand in inkscape

def oh_2019_kim_2015(fignum=2, show=True):
    """Reproduces the results from Oh_Minimizing_2019 and kim_neural_2015 and plots
    them in a format similar to their plots.

    An empty subplot is created to manually add the figure from their
    paper.

    """
    context_color = np.ones(3) * 0.5
    figsize = (6, 6)
    fig, axes = plt.subplots(3, 2, clear=True, num=fignum, figsize=figsize)
    # fig = plt.figure(figsize=figsize, clear=True, num=fignum)
    # axes_00 = fig.add_subplot(3, 2, 1)
    # axes_01 = fig.add_subplot(3, 2, 2, sharey=axes_00)
    # axes_10 = fig.add_subplot(3, 2, 3)
    # axes_11 = fig.add_subplot(3, 2, 4, sharex=axes_10, sharey=axes_10)
    # axes_20 = fig.add_subplot(3, 2, 5, sharex=axes_10)
    # axes_21 = fig.add_subplot(3, 2, 6, sharex=axes_10, sharey=axes_20)
    # axes = np.array([[axes_00, axes_01],
    #                  [axes_10, axes_11],
    #                  [axes_20, axes_21]])

    # a)
    task_kim, agent_pars = sims.kim_2015(plot=False)
    agent_kim = model.LRMeanSD(**agent_pars)
    pandata, pandagent, _ = thh.run(agent_kim, pars=task_kim)
    pandota_kim = thh.join_pandas(pandata, pandagent)
    contexts = pandota_kim['ix_context'].values
    contexts[contexts == pars.CLAMP_INDEX] = 0
    contexts[contexts == 1] = -40
    contexts[contexts == 2] = 40
    axes[0, 0].plot(contexts, color=context_color)
    axes[0, 1].plot(contexts, color=context_color)
    axes[0, 0].plot(-pandota_kim['pos(t)'] - pandota_kim['action'])
    axes[0, 0].set_yticks([-40, 0, 40])
    axes[0, 0].set_ylabel('Adaptation')
    axes[0, 0].set_title('Simulations')
    axes[0, 1].set_title('Experimental data')
    xlim = axes[0, 0].get_xlim()
    ylim = axes[0, 0].get_ylim()
    axes[0, 1].set_xlim(xlim)
    axes[0, 1].set_ylim(ylim)

    # b)
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
    axes[1, 0].plot(contexts, color=context_color)
    axes[1, 1].plot(contexts_20, color=context_color)
    axes[1, 0].plot(-pandota_20['pos(t)'] - pandota_20['action'])

    axes[1, 0].set_ylabel('Adaptation')
    xlim = axes[1, 0].get_xlim()
    ylim = axes[1, 0].get_ylim()
    ylim = (-5, ylim[1])
    axes[1, 0].set_ylim(ylim)
    axes[1, 1].set_xlim(xlim)
    axes[1, 1].set_ylim(ylim)

    # c)
    agent = model.LRMeanSD(**agent_pars)
    agent.all_learn = True
    pandata, pandagent, _ = thh.run(agent, pars=task_10)
    pandota_10 = thh.join_pandas(pandata, pandagent)
    contexts = pandota_10['ix_context'].values
    contexts[contexts == pars.CLAMP_INDEX] = 0
    contexts[contexts == 1] = 10
    contexts_10 = contexts
    axes[2, 0].plot(contexts_10, color=context_color)
    axes[2, 1].plot(contexts_10, color=context_color)
    axes[2, 0].plot(-pandota_10['pos(t)'] - pandota_10['action'])
    axes[2, 0].set_xlabel('Trials')
    axes[2, 0].set_ylabel('Adaptation')
    xlim = axes[2, 0].get_xlim()
    ylim = axes[2, 0].get_ylim()
    ylim = (-3, ylim[1])
    axes[2, 0].set_ylim(ylim)
    axes[2, 1].set_xlim(xlim)
    axes[2, 1].set_ylim(ylim)

    axes[2, 1].set_xlabel('Trials')

    for idsw, (sw1, sw2) in enumerate(zip(switches, switches[1:])):
        number = np.ceil((idsw + 1) / 2).astype(int)
        letter = '$O_{}$' if idsw % 2 == 0 else r'$A_{}$'
        label = letter.format(number)
        x_pos = (sw2 + sw1) / 2
        axeses = axes[1:, :].reshape(-1)
        for axis in axeses:
            y_pos = axis.get_ylim()[-1] * 0.88
            axis.text(s=label, x=x_pos, y=y_pos, ha='center',
                      fontsize=8)

    # Subplot labels
    axes[0, 0].text(x=-0.1, y=1, s='A',
                    transform=axes[0, 0].transAxes,
                    fontdict={'size': 14})
    axes[1, 0].text(x=-0.1, y=1, s='B',
                    transform=axes[1, 0].transAxes,
                    fontdict={'size': 14})
    axes[2, 0].text(x=-0.1, y=1, s='C',
                    transform=axes[2, 0].transAxes,
                    fontdict={'size': 14})
    for axis in axes[:, 1].reshape(-1):
        axis.set_yticks([])
    for axis in axes[1, :].reshape(-1):
        axis.set_xticks([])
    fig.tight_layout()
    fig.align_ylabels(axes[:, 0])
    plt.savefig(FIGURE_FOLDER + 'figure_{}.png'.format(fignum), dpi=600)
    plt.savefig(FIGURE_FOLDER + 'figure_{}.svg'.format(fignum), format='svg')
    if show:
        plt.draw()
        plt.show(block=False)


def herzfeld_2014(fignum=3, show=True):
    """Reproduces the results from Herzfeld et al 2014.

    """
    figsize = (4, 5)
    colors = {'high': np.array((95, 109, 212)) / 256,
              'low': np.array((212, 198, 95)) / 256}
    optimal = -13
    repeats = 300  # participants * (number of blocks)
    num_trials = 30
    fig = plt.figure(clear=True, num=fignum, figsize=figsize)
    fig, axes = plt.subplots(2, 1, squeeze=True, clear=True,
                             num=fignum, sharex=True)

    agent = {}
    data = {'high': [], 'low': []}
    for idx in range(repeats):
        outs = sims.herzfeld_2014(num_trials=num_trials, plot=False)
        tasks = outs[:2]
        agent_pars = outs[2:]

        for name, t_pars, a_pars in zip(['high', 'low'], tasks, agent_pars):
            agent[name] = model.LRMeanSD(**a_pars)
            agent[name].all_learn = True
            agent[name].threshold_learn = 0.2
        for name, c_pars in zip(['high', 'low'], tasks):
            pandata, pandagent, _ = thh.run(agent[name], pars=c_pars)
            pandota = thh.join_pandas(pandata, pandagent)
            data[name].append(pandota)
    data['high'] = pd.concat(data['high'])  # .groupby('trial').mean()
    data['low'] = pd.concat(data['low'])  # .groupby('trial').mean()
    data['high'].reset_index('trial', inplace=True)
    data['low'].reset_index('trial', inplace=True)
    all_data = pd.concat([data['high'], data['low']], keys=['high', 'low'],
                         names=['eta'])
    all_data.reset_index('eta', inplace=True)
    all_data['mag_mu_1'] /= optimal
    sns.lineplot(data=all_data, x='trial', y='mag_mu_1',
                 hue='eta', ax=axes[0], palette=colors)
    axes[0].legend(title='')
    axes[0].set_ylabel('Adaptation (% of optimal)')
    axes[0].set_title('Learning from error')
    axes[0].text(x=-0.1, y=1, s='A',
                 transform=axes[0].transAxes,
                 fontdict={'size': 14})

    # Learning from error
    sensitive = [[], []]
    flag_flip = True
    for idx, eta in enumerate(['low', 'high']):
        error = data[eta]['hand'].values[1:] + 0.01
        learning = np.diff(data[eta]['mag_mu_1'])
        if flag_flip:
            neg_error = error < 0
            learning[neg_error] *= -1
            error = np.abs(error)
            learning = np.abs(learning)
        error_sensitivity = learning / error / (-optimal)
        sensitive[idx] = pd.DataFrame({'error': error,
                                       'sensi': error_sensitivity,
                                       'z_label': eta,
                                       'trial': data[eta]['trial'].values[1:]})
    sensitive_panda = pd.concat(sensitive)
    bins = np.concatenate([np.arange(-30, -5), np.arange(-5, 5, 0.1),
                           np.arange(5, 30)])
    sensitive_panda['bin_error'] = pd.cut(sensitive_panda['error'],
                                          bins=bins,
                                          labels=bins[:-1])
    trial_bins = np.linspace(0, num_trials, 7, endpoint=True)
    sensitive_panda['bin_trial'] = pd.cut(sensitive_panda['trial'],
                                          bins=trial_bins,
                                          labels=trial_bins[:-1])
    sns.lineplot(data=sensitive_panda, y='sensi', x='bin_trial',
                 hue='z_label', palette=colors, ax=axes[1],
                 style='z_label', markers=True, dashes=False)
    axes[1].set_xlabel('Trials')
    axes[1].set_ylabel('Sensitivity\nto error (a.u.)')
    yticks = axes[1].get_yticks()
    axes[1].set_yticks(yticks[[0, -1]])
    axes[1].set_yticklabels([0, ''])
    axes[1].get_legend().remove()
    axes[1].set_title('Error sensitivity')
    axes[1].text(x=-0.1, y=1, s='B',
                 transform=axes[1].transAxes,
                 fontdict={'size': 14})

    fig.tight_layout()

    plt.savefig(FIGURE_FOLDER + 'figure_{}.png'.format(fignum), dpi=600)
    plt.savefig(FIGURE_FOLDER + 'figure_{}.svg'.format(fignum), format='svg')
    if show:
        plt.draw()
        plt.show(block=False)


def davidson_2004(fignum=4, show=True):
    """Reproduces the results from Davidson_Scaling_2004, leaving an empty
    subplot to put in the results from their paper.

    """
    repeats = 8  # No. of participants per group
    figsize = (4, 3)
    colors = {'-8': np.array((95, 109, 212)) / 256,
              '+8': np.array((212, 198, 95)) / 256}
    ran = [161, 201]

    fig, axes = plt.subplots(1, 2, num=fignum, clear=True, squeeze=True,
                             figsize=figsize, sharex=True, sharey=True)
    task_m8, task_p8, agent_pars_m8, agent_pars_p8 = sims.davidson_2004(plot=False)

    agent_m8 = model.LRMeanSD(**agent_pars_m8)
    agent_p8 = model.LRMeanSD(**agent_pars_p8)

    tasks = [task_m8, task_p8]
    agents = [agent_m8, agent_p8]
    names = ['-8', '+8']
    data = {name: [] for name in names}
    for idx, (task, agent, name) in enumerate(zip(tasks, agents, names)):
        for idx in range(repeats):
            pandata, pandagent, _ = thh.run(agent, pars=task)
            pandota = thh.join_pandas(pandata, pandagent)
            data[name].append(pandota)
        data[name] = pd.concat(data[name])
        data[name]['pos(t)'] = np.abs(data[name]['pos(t)'])
    data = pd.concat(data, axis=0, names=['Group', 'trial'])
    data.reset_index('Group', inplace=True)
    sns.lineplot(data=data, x='trial', y='pos(t)', ax=axes[0],
                 hue='Group', palette=colors, ci='sd')
    axes[0].set_xlim(ran)
    ticks = np.array(axes[0].get_xticks(), dtype=int) - ran[0] + 1
    axes[0].set_xticklabels(ticks)
    # axes[0].plot(np.arange(ran[1] - ran[0] + 1), error[name], color=colors[idx],
    #              label=name)
    # axes[0].legend()
    axes[0].set_xlabel('Trials after switch')
    axes[1].set_xlabel('Trials after switch')
    axes[0].set_ylabel('Error (a.u.)')
    axes[0].text(x=-0.1, y=1.05, s='A',
                 transform=axes[0].transAxes,
                 fontdict={'size': 14})
    axes[1].text(x=-0.1, y=1.05, s='B',
                 transform=axes[1].transAxes,
                 fontdict={'size': 14})

    fig.tight_layout()
    plt.savefig(FIGURE_FOLDER + 'figure_{}.png'.format(fignum), dpi=600)
    plt.savefig(FIGURE_FOLDER + 'figure_{}.svg'.format(fignum), format='svg')
    if show:
        plt.draw()
        plt.show(block=False)


def vaswani_2013(fignum=5, show=True, pandota=None):
    """Reproduces the results from Vaswani_Decay_2013, especifically their
    figures 2a-c.

    """
    reps = 10
    figsize = (6, 7)
    colors = ['blue', 'green', 'red', 'c']
    context_colors = ['black', 'tab:orange', 'tab:purple', 'tab:brown',
                      'tab:olive']
    mags = gs.GridSpec(3, 4, height_ratios=[0.4, 0.2, 0.2])
    fig = plt.figure(num=fignum, clear=True, figsize=figsize)
    axes_con = [fig.add_subplot(mags[1, 0])]
    axes_con.append(fig.add_subplot(mags[1, 1], sharey=axes_con[0]))
    axes_con.append(fig.add_subplot(mags[2, 0], sharey=axes_con[0]))
    axes_con.append(fig.add_subplot(mags[2, 1], sharey=axes_con[0]))
    axes_sum = [fig.add_subplot(mags[0, 0:2]), ]
    axes_sum.append(fig.add_subplot(mags[0, 2:4], sharex=axes_sum[0]))
    axes_lag = fig.add_subplot(mags[1:3, 2:4])
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
            all_pandas.append(c_pandota)
    pandota = pd.concat(all_pandas, axis=0)
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
                         ax=axes_con[idx], ci='sd')
    for idx in range(4):
        axes_con[idx].set_title('Group {}'.format(names[idx]))
        axes_con[idx].set_xlabel('')
        axes_con[idx].set_ylabel('')
        axes_con[idx].set_xticks([])
        axes_con[idx].set_yticks([])
        axes_con[idx].set_ylim([0, 1])
    axes_con[0].text(x=-0.3, y=1.1, s='C',
                     transform=axes_con[0].transAxes,
                     fontdict={'size': 14})
    axes_con[0].set_yticks([0, 1])

    # Plot summary adaptation:
    condi = 'trial >= 75 and trial <= 175 and group != 1.4'
    pandota_e = pandota.query(condi)
    pandota_e.reset_index('trial', inplace=True)
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
    axes_sum[1].set_yticks([])
    axes_sum[0].set_xlabel('Trials since start of error-clamp')
    axes_sum[1].set_xlabel('Trials since start of error-clamp')
    axes_sum[0].set_ylabel('Adaptation index')

    axes_sum[0].text(x=-0.1, y=1, s='A',
                     transform=axes_sum[0].transAxes,
                     fontdict={'size': 14})
    axes_sum[1].text(x=-0.1, y=1, s='B',
                     transform=axes_sum[1].transAxes,
                     fontdict={'size': 14})

    # Plot lags
    panda_lag = pandota.query('trial > 90 and trial <= 140')
    panda_lag.reset_index('trial', inplace=True)
    panda_lag.loc[:, 'trial'] -= 100
    axes_lag.set_ylabel('p(ctx)')
    axes_lag.text(x=-0.1, y=1.033, s='D',
                  transform=axes_lag.transAxes,
                  fontdict={'size': 14})

    # for run in np.unique(pandota['run']):
    #     datum = panda_lag.query('run == @run')
    #     group_label = datum.iloc[0]['group']
    #     c_group = int(group_label * 10 - 1.1 * 10)
    #     if c_group == 3:
    #         continue
    #     color = colors[c_group]
    #     axes_lag.plot(datum['trial'], datum['con1'],
    #                   color=color, alpha=0.8)
    sns.lineplot(data=panda_lag, x='trial', y='con1',
                 hue='group', units='run', estimator=None,
                 palette=colors)
    axes_lag.set_xlabel('Trials since start of error-clamp')
    mags.tight_layout(fig)

    # Kidnapped from top to run after tight layout
    axes_con[2].set_xticks([0, 100])
    axes_con[2].set_xlabel('Trial')
    axes_con[0].set_ylabel('p(ctx)')

    plt.savefig(FIGURE_FOLDER + 'figure_{}.png'.format(fignum), dpi=600)
    plt.savefig(FIGURE_FOLDER + 'figure_{}.svg'.format(fignum), format='svg')

    if show:
        plt.draw()
        plt.show(block=False)
