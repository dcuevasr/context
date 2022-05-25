# -*- coding: utf-8 -*-
# ../code/heald.py

"""Some functions to test things for the Heald model and related stuff."""
from itertools import product

import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt
import arviz as az

plt.ion()


def i_want_to_print_yes():
    """yes"""
    print('yes')


def priors_over_x(obs=None, coin_args=None, mine_args=None):
    """Check the priors over x_t and compare (a,b) to (mu, sd)."""
    model_coin = pm.Model(name='COIN')
    model_mine = pm.Model(name='Mine')

    if coin_args is not None:
        sigma_a, sigma_d, mu_a, sigma_q = coin_args
        mu_b = 0
    else:
        sigma_a = 2
        sigma_d = 2
        mu_a = 0
        mu_b = 0  # Fixed
        sigma_q = 1
    with model_coin:
        ab = pm.MvNormal('ab', mu=np.array([mu_a, mu_b]),
                         cov=np.diag([sigma_a, sigma_d]), shape=2)
        bound_a = pm.Deterministic('a', pm.math.sigmoid(ab[0]))
        mu_x = pm.Deterministic('mu_x', ab[1] / (1 - bound_a))
        sd_x = pm.Deterministic('sd_x', sigma_q / (1 - bound_a ** 2))
        if obs is None:
            _ = pm.Normal('x', mu=mu_x, sd=sd_x)
        else:
            _ = pm.Normal('x', mu=mu_x, sd=sd_x, observed=obs)

    if mine_args is not None:
        mu_x, nu_x, alpha, beta = mine_args
    else:
        mu_x = 0
        nu_x = 1
        alpha = 36 / 2
        beta = 10 / 2
    with model_mine:
        sd = pm.Gamma('sd_x', alpha=alpha, beta=beta)
        mu = pm.Normal('mu_x', mu=mu_x, sd=sd / nu_x)
        if obs is None:
            _ = pm.Normal('x', mu=mu, sd=sd)
        else:
            _ = pm.Normal('x', mu=mu, sd=sd, observed=obs)

    return model_coin, model_mine


def sample_models(models):
    """Samples from the given pymc3 models. Returns the samples"""
    num_models = len(models)
    samples = [[]] * num_models
    for ix_mo, model in enumerate(models):
        with model:
            samples[ix_mo] = pm.sample(draws=3000, cores=8)

    return samples


def plot_samples(samples, fignum=1):
    """Plots scatters of the samples"""
    num_models = len(samples)
    fig, axes = plt.subplots(num_models, 1, num=fignum, clear=True,
                             sharex=True, sharey=True)

    for ix_sam, sample in enumerate(samples):
        model_name = sample.varnames[0].split('_')[0]
        mu_name = model_name + '_mu_x'
        sd_name = model_name + '_sd_x'
        mus = sample.get_values(mu_name)
        sds = sample.get_values(sd_name)
        axes[ix_sam].scatter(mus, sds, alpha=0.1)
        axes[ix_sam].set_xlabel(r'$\mu$')
        axes[ix_sam].set_ylabel(r'$\sigma$')
    return fig, axes


def jointplot_samples(samples, fignum=2):
    """Plots the posterior joint plot."""
    fig, axes = plt.subplots(
        2, 1, num=fignum, clear=True, sharex=True, sharey=True)
    for ix_samples, sample in enumerate(samples):
        model_name = sample.varnames[0].split('_')[0]
        mux = model_name + '_mu_x'
        sdx = model_name + '_sd_x'
        az.plot_pair(sample, var_names=[mux, sdx], kind='kde',
                     ax=axes[ix_samples])
    return fig, axes


def from_mod_to_plot():
    models = priors_over_x()
    samples = sample_models(models)
    plot_samples(samples)
    jointplot_samples(samples)
    return samples, models


def infer_one_from_other(model_name, samples=None):
    """Infers the parameters of the --model_name-- ('mine' or 'coin') using the
    --samples--. If no samples are provided, they are obtained with the other
    model."""
    ix_model = int(model_name == 'mine')
    name_prefix = 'Mine' if not ix_model else 'COIN'
    if samples is None:
        model = priors_over_x()[int(not ix_model)]
        name_prefix = model.name
        samples = sample_models([model])[0]
    observed_name = '_'.join([name_prefix, 'x'])
    infer_model = priors_over_x(
        obs=samples.get_values(observed_name))[ix_model]
    return infer_model


def confucius_matrix(samples, kwargs_priors=None,
                     plot=False, fignum=3):
    """Builds a confusing matrix, where data is generated with the two models
    and both models are used to infer each data set. The model evidence
    (the elbow, really) is used to plot the confused matrix.

    Note that inference is done with ADVI, to be able to collect the elbow.

    """
    pre_models = priors_over_x(**kwargs_priors)
    mean_fields = np.empty((len(pre_models), len(samples)), dtype=object)
    elbows = np.zeros(mean_fields.shape)
    for ix_sam, sam in enumerate(samples):
        models = priors_over_x(obs=sam, **kwargs_priors)
        for ix_mod, mod in enumerate(models):
            mean_fields[ix_mod, ix_sam] = pm.fit(method='advi', model=mod)
            elbows[ix_mod, ix_sam] = mean_fields[ix_mod, ix_sam].hist[-1]
    if plot:
        plot_elbows(elbows, fignum=fignum)
    return mean_fields, elbows


def plot_elbows(elbows, axis=None, fignum=3):
    """Plots the confession matrix for the given elbows.

    """
    if axis is None:
        fig, axis = plt.subplots(1, 1, clear=True, num=fignum)
    showy = axis.imshow(elbows)
    axis.set_xticks(np.arange(elbows.shape[0]) - 0.5)
    axis.set_yticks(np.arange(elbows.shape[1]) - 0.5)
    axis.set_xticks(np.arange(elbows.shape[0]),
                    labels=['COIN', 'Reduced'], minor=True)
    axis.set_yticks(np.arange(elbows.shape[0]),
                    labels=['COIN', 'Reduced'], minor=True)
    plt.colorbar(showy, ax=axis, ticks=[elbows.min(), elbows.max()],
                 label='Model evidence')
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.set_ylabel('Models')
    axis.set_xlabel('Data sets')
    axis.grid(visible=True, linewidth=3, color='white')


def generate_elbows_crossed():
    """Generates all the elbows, which are all combinations of generating
    samples and fitting them with all the model parameters in _model_hypers.

    This takes a long time to run (~1h?).
    """
    coin_pars, mine_pars = _model_hypers()
    num_sets = coin_pars.shape[0]
    samples = np.empty((num_sets, 2), dtype=object)
    for ix_set in range(num_sets):
        kwargs = {'coin_args': coin_pars[ix_set, ...],
                  'mine_args': mine_pars[ix_set, ...]}
        c_models = priors_over_x(**kwargs)
        c_samples = sample_models(c_models)
        samples[ix_set, :] = c_samples

    elbows = np.zeros((num_sets, num_sets, 2, 2))
    for ix_sams, sams in enumerate(samples):
        for ix_set in range(num_sets):
            kwargs = {'coin_args': coin_pars[ix_set, ...],
                      'mine_args': mine_pars[ix_set, ...]}
            _, c_elbows = confucius_matrix(samples=sams,
                                           kwargs_priors=kwargs)
            elbows[ix_set, ix_sams, ...] = c_elbows


def _flatten_elbows(elbows):
    """Plots the confusion matrix generated using the elbows from
    priors_model_evidence.

    """
    num_sets = elbows.shape[0]
    flat_elbows = np.zeros((2 * num_sets, 2 * num_sets))
    for ix_row in range(num_sets):
        for ix_col in range(num_sets):
            flat_elbows[2 * ix_row:(2 * ix_row + 2), 2 * ix_col:(2 * ix_col + 2)] = \
                elbows[ix_row, ix_col, ...]
    flat_elbows /= flat_elbows.max(axis=0)[None, :]
    return flat_elbows


def _model_hypers():
    """Values I selected to run the simulations and inference for
    priors_model_evidence.

    """
    # For COIN
    sigma_a_all = [1, 5]
    sigma_b_all = [1, 5]
    mu_a_all = [0, 2]
    sigma_q_all = [1, 5]
    coin_combs = np.array(list(product(sigma_a_all, sigma_b_all, mu_a_all,
                                       sigma_q_all)))

    # For mine
    mu_x = [0, 2]
    nu_x = [1, 10]
    alpha = [5, 15]
    beta = [2, 6]
    mine_combs = np.array(list(product(mu_x, nu_x, alpha, beta)))

    return coin_combs, mine_combs


def generate_elbows_normal():
    """Generates data with a Gaussian with some random values for mu and sd,
    and generates the elbows with the models as parametrized via
    _model_hypers().

    """
    num_samples = 1000
    mus = np.linspace(-2, 2, 9, endpoint=True)
    sds = np.linspace(0.1, 5, 9)

    samples = []
    for mu, sd in zip(mus, sds):
        c_samples = mu + sd * np.random.normal(size=num_samples)
        samples.append(c_samples)

    coin_pars, mine_pars = _model_hypers()
    num_sets = coin_pars.shape[0]
    elbows = np.zeros((num_sets, num_sets, 2, 2))
    for ix_sams, sams in enumerate(samples):
        for ix_set in range(num_sets):
            kwargs = {'coin_args': coin_pars[ix_set, ...],
                      'mine_args': mine_pars[ix_set, ...]}
            _, c_elbows = confucius_matrix(samples=sams,
                                           kwargs_priors=kwargs)
            elbows[ix_set, ix_sams, ...] = c_elbows
    return elbows


def generate_bfs_normal():
    """Generates the Bayes factors.
    """
    num_samples = 1000
    num_mcmc = 1000
    num_sampled = 2
    mus = np.linspace(-2, 2, num_sampled, endpoint=True)
    sds = np.linspace(0.1, 5, num_sampled)

    samples = []
    for mu, sd in zip(mus, sds):
        c_samples = mu + sd * np.random.normal(size=num_samples)
        samples.append(c_samples)

    coin_pars, mine_pars = _model_hypers()
    num_sets = coin_pars.shape[0]
    comps = np.zeros((num_sampled, num_sets), dtype=object)
    for ix_sams, sams in enumerate(samples):
        for ix_set in range(num_sets):
            kwargs = {'coin_args': coin_pars[ix_set, ...],
                      'mine_args': mine_pars[ix_set, ...]}
            models = priors_over_x(obs=samples, **kwargs)
            coin = pm.sample(model=models[0], draws=num_mcmc)
            mine = pm.sample(model=models[1], draws=num_mcmc)
            c_comp = pm.compare({'mine': mine, 'coin': coin})
            comps[ix_sams, ix_set] = c_comp
    return comps
