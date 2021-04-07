# -*- coding: utf-8 -*-
# ./adaptation/model.py

import numpy as np
from scipy import stats
import pandas as pd
from matplotlib import pyplot as plt

from pars import agent as pars

"""Bayesian motor adaptation model."""


class LeftRightAgent(object):
    """Subclass that knows two contexts(plus baseline): one pointing left and one
    pointing right. The magnitude of these forces are inferred during the
    experiment.

    The class takes many values from the pars.py module, which must be
    importable and must contain a dictionary called --agent--. For the default
    values, see pars.py.

    The necessary entries of the dictionary are the following:
    'action_sd' : float
    Std for the Gaussian from which actions should be sampled.

    'obs_noise' : float
    Assumed Std for the noise in the observations.

    'force_sd' : float
    Assumed Std for the force applied by the environment

    'prediction_noise' : float
    Inverse precision given to the predictions of the current state given the
    last state and the last action. See self.update_magnitudes for more.

    'reset_after_change' : bool
    Whether to reset the priors over context when self.reset() is called by
    external actors (i.e. the task).

    'max_force': float
    Maximum force (in units of distance (Newtons times delta t)). See
    the model implementation in the notes.pdf file for an explanation of this.
    """
    name = 'LR'
    # Free parameters with default values. Can be overwritten by __init__
    action_sd = pars['action_sd']  # SD of a Gaussian for action uncertainty
    cue_noise = 0.1  # If ix_cue is observed, the posterior over the
                     # corresponding context is 1 - 2 * cue_noise.
    obs_sd = pars['obs_noise']
    angles = np.array([0, 0.01, -0.01])
    force_sds = pars['force_sd']  # np.array([0.01, 0.2, 0.2])
    prediction_noise = pars['prediction_noise']
    reset_after_change = pars['reset_after_change']

    is_reset = True  # Whether a new miniblock started

    preferred = 1

    sample_context_mode = 'mode'
    sample_force_mode = 'mode'

    max_force = pars['max_force']  # Maximum force the agent can exert

    context_noise = 0.0001

    def __init__(self, obs_sd=None, action_sd=None, cue_noise=None,
                 angles=None, context_noise=None, prediction_noise=None,
                 reset_after_change=None, force_sds=None):
        """Initializes the known left and right contexts, as well as
        baseline.

        """
        if obs_sd:
            self.obs_sd = obs_sd
        if action_sd:
            self.action_sd = action_sd
        if cue_noise:
            self.cue_noise = cue_noise
        if angles:
            self.angles = np.array(angles)
        if context_noise:
            self.context_noise = context_noise
        if prediction_noise:
            self.prediction_noise = prediction_noise
        if reset_after_change:
            self.reset_after_change = reset_after_change
        if force_sds is not None:
            self.force_sds = force_sds

        self.num_contexts = 3
        _, self.magnitudes = self.__init_contexts()
        self.magnitude_history = [self.magnitudes]
        self.decision_history = []
        self.hand_position_history = [0]
        self.hand_position = 0
        self.action_history = [0]
        self.log_context = np.log([0.8, 0.1, 0.1])
        self.log_context_history = [self.log_context]

        self.sample_norm = stats.norm().rvs

    def __init_contexts(self, ):
        r"""Creates baseline, left and right contexts, with a known angle (no force,
        right, left), as well as the hyperparameters for the magnitudes,
        assumed to be Gaussians with mu and sd.

        """
        magnitudes = np.hstack([self.angles[:, None],
                               self.force_sds[:, None]])
        return self.angles, magnitudes

    def reset(self, priors=None):
        """Resets the priors over context to a uniform distribution.
        If --priors-- are provided, they are used instead.

        """
        self.is_reset = True
        if priors is None:
            self.log_context = - np.log(self.num_contexts) * np.ones(self.num_contexts)
        else:
            if not np.isclose(np.sum(priors), 1):
                raise ValueError('Provided priors do not add to 1')
            self.log_context = np.log(priors)

    def make_decision(self, ):
        """Makes a decision after all the inferences for the trial have been
        carried out.

        The logic is to choose an action such that
        p(pos) - p(force) + p(s_(t+1)| action) = 0

        A simple proxy is to use the mode of each one of this, which is what
        this function does for now.

        The decision is returned and also saved into self.decision_history.

        """
        c_pos = self.hand_position
        c_context = self.sample_context('mode')
        c_force = self.magnitudes[c_context][0]
        action_mu = -c_pos - c_force
        action = self.sample_norm() * self.action_sd + action_mu
        if abs(action) > self.max_force:
            action = action / abs(action) * self.max_force
        self.action = action
        self.action_history.append(action)
        return action

    def update_magnitudes(self, ):
        """At the beginning of a trial, updates the inferred magnitudes for
        the contexts given the previous action and its outcome.

        Saves the results to self.inferred_magnitude_pars and samples (TODO: or
        uses the mean) to save in self.context_pars.

        Parameters
        ----------
        likelihood_pars : iterable size=(2,)
        Mean and standard deviation of the current observation, used as the
        likelihood to update the posteriors over magnitudes.

        """
        ix_context = self.sample_context(t=-2)
        mu_p, sd_p = self.magnitudes[ix_context]
        mu_l, sd_l = self.predict_outcome()[1][ix_context]
        mu_l = mu_p + self.hand_position - mu_l
        mu_post = (mu_l * sd_p ** 2 + mu_p * sd_l ** 2) / \
            (sd_l ** 2 + sd_p ** 2)
        sd_post = np.sqrt((sd_l ** 2 * sd_p ** 2) / (sd_p ** 2 + sd_l ** 2))
        c_con = np.array(self.magnitudes)
        c_con[ix_context] = [mu_post, sd_post]
        self.magnitudes = c_con
        self.magnitude_history.append(self.magnitudes)

    def predict_outcome(self, hand_position=None, action=None):
        """Given an action, generates a probability distribution over all
        possible outcomes.

        If --hand_position-- and/or --action-- are not provided, the values
        are taken from self.xxx_history[-1/-2] (-2 because one_trial calls
        this function after having saved the current hand position, but before
        saving the current action).

        Returns
        -------
        predicted_hand : function
        Function with two inputs (hand_position, context_index) that returns
        the log-likelihood of the hand_position given the context.

        force_pars : list (of stats.norm.pdf instances)
        List containing the posterior over final position after the action
        has been taken. force_pars[ix_context](x) returns the posterior
        probability for x given context ix_context.

        """
        if action is None:
            action = self.action_history[-1]
        action_mu = action
        action_sd = self.action_sd
        if hand_position is None:
            hand_position = self.hand_position_history[-2]
        funs = []
        posterior_pars = []
        for ix_context in range(self.num_contexts):
            force_mag_mu, force_mag_sd = self.magnitudes[ix_context]
            pos_mu = force_mag_mu + action_mu + hand_position
            pos_sd = np.sqrt(force_mag_sd ** 2 + action_sd ** 2 +
                             self.obs_sd ** 2)
            posterior_pars.append([pos_mu, pos_sd])
            fun = stats.norm(loc=pos_mu, scale=pos_sd).logpdf
            funs.append(fun)

        def predicted_hand(pos, context):
            return funs[context](pos)
        return predicted_hand, posterior_pars

    def infer_context(self, hand_position, cue=None):
        """Infers the current context given the current observation, as well
        as the previous action and outcome.

        """
        if not self.is_reset:
            lh_fun, lh_pars = self.predict_outcome()
            # Hand position likelihood:
            log_li_hand = np.array([lh_fun(hand_position, ix_context)
                                    for ix_context in range(self.num_contexts)])
            li_hand = np.exp(log_li_hand - log_li_hand.max())
            p_hand = li_hand / li_hand.sum()
            p_hand += self.prediction_noise
            p_hand /= p_hand.sum()
            log_li_hand = np.log(p_hand)
        else:
            log_li_hand = -np.log(self.num_contexts) * np.ones(self.num_contexts)

        # Cue likelihood:
        if cue is None:
            log_cue = np.zeros(self.num_contexts)
        else:
            cue_li = self.cue_noise * np.ones(self.num_contexts)
            cue_li[cue] = 1 - (self.num_contexts - 1) * self.cue_noise
            log_cue = np.log(cue_li)
        prior = np.exp(self.log_context - self.log_context.max()) + \
            self.context_noise
        log_prior = np.log(prior / prior.sum())
        full_logli = log_li_hand + log_cue + log_prior
        full_li = np.exp(full_logli - full_logli.max())
        full_li /= full_li.sum()
        full_logli = np.log(full_li)
        self.log_context = full_logli
        self.log_context_history.append(full_logli)

    def sample_context(self, mode=None, t=-1):
        """This function defines the logic to obtain a single number from
        the current estimation of the current context. Implemented logics
        are 'mode', 'sample'.

        Parameters
        ----------
        mode : str
        One of {'mode', 'sample'}. If None, it will
        be taken from self.sample_context_mode.

        Returns
        -------
        sampled_context : int
        Integer to index the context.

        """
        if mode is None:
            mode = self.sample_context_mode
        log_context = self.log_context_history[t]
        posti = np.exp(log_context - log_context.max())
        if mode == 'mode':
            sampled_context = np.argmax(posti)
        elif mode == 'sample':
            sampled_context = np.random.choice(np.arange(len(posti)),
                                               p=posti)
        return sampled_context

    def sample_force(self, mode=None):
        """This function defines the logic to obtain a single number from
        the current estimation of the force. Implemented logics
        are 'mean', 'sample'.

        Parameters
        ----------
        mode : str
        One of {'mean', 'sample'}. If None, it will
        be taken from self.sample_context_mode.

        Returns
        -------
        sampled_context : int
        Integer to index the context.

        """
        if mode is None:
            mode = self.sample_force_mode
        ix_context = self.sample_context()
        force_pars = self.force_pars[ix_context]
        if mode == 'mean':
            sampled_force = force_pars[0]
        elif mode == 'sample':
            sampled_force = stats.norm(loc=force_pars[0],
                                       scale=force_pars[1]).rvs()
        return sampled_force

    def pandify_data(self):
        """Pandifies the history of the agent."""
        context = np.exp(self.log_context_history)
        max_context = np.argmax(context, axis=1)
        magnitudes = np.array(self.magnitude_history)
        mag_mu = magnitudes[..., 0]
        mag_sd = magnitudes[..., 1]
        hand = self.hand_position_history
        actions = self.action_history
        trial_number = np.arange(context.shape[0]) - 1
        aggregate = np.stack([*context.T, max_context, *mag_mu.T,
                              *mag_sd.T,
                              hand,
                              actions,
                              trial_number],
                             axis=1)
        pandata = pd.DataFrame(aggregate,
                               columns=['con0', 'con1', 'con2',
                                        'con_t',
                                        'mag_mu_0', 'mag_mu_1', 'mag_mu_2',
                                        'mag_sd_0', 'mag_sd_1', 'mag_sd_2',
                                        'hand', 'action', 'trial'])
        pandata.reset_index(drop=True, inplace=True)
        pandata.set_index('trial', inplace=True)
        return pandata

    def one_trial(self, hand_position, cue=None):
        r"""Processes the entire trial (\Delta t): receives an observation (hand
        position and contextual cue, if available), updates context and force
        inference, makes a decision.

        """
        # if len(self.hand_position_history) >= 12:
        #     ipdb.set_trace()
        self.hand_position = hand_position
        self.hand_position_history.append(hand_position)
        self.infer_context(hand_position, cue)
        self.update_magnitudes()
        action = self.make_decision()
        self.is_reset = False
        return action

    def plot_mu(self, trials=None, fignum=1, axis=None):
        """Plots the adaptation (mu of the force) as a function of time."""
        colors = ['black', 'red', 'blue']
        if axis is None:
            fig, axis = plt.subplots(1, 1, num=fignum, clear=True)
        mag_hist = np.array(self.magnitude_history)
        for ix_context in range(self.num_contexts):
            axis.plot(mag_hist[:, ix_context, 0],
                      color=colors[ix_context])
        axis.set_xlabel('Trial')
        axis.set_ylabel('Adaptation')

    def plot_contexts(self, offset=0, height=1, alpha=1, fignum=2, axis=None):
        """Plots a stacked bar plot representing the posterior over
        contexts for every trial.

        """
        con_t = np.exp(np.array(self.log_context_history) -
                       np.array(self.log_context_history).max(axis=1)[:, None])
        con_t = con_t / con_t.sum(axis=1)[:, None] * height + offset
        trials = np.arange(con_t.shape[0])
        colors = ['black', 'red', 'blue']
        if axis is None:
            fig, axis = plt.subplots(1, 1, num=fignum, clear=True)
        for ix_con in range(self.num_contexts):
            axis.plot(trials, con_t[:, ix_con], color=colors[ix_con],
                      alpha=alpha)

    def plot_position(self, alpha=0.9, fignum=3, axis=None):
        """Plots the observation of the in time."""
        obs = np.array(self.hand_position_history)
        if axis is None:
            fig, axis = plt.subplots(1, 1, num=fignum, clear=True)
        axis.plot(obs, alpha=alpha)

    def plot_full(self, fignum=4):
        """Plots EVERYTHING (cue Gary Oldman in Leon). """
        fig, axis = plt.subplots(1, 1, num=fignum, clear=True)
        self.plot_contexts(offset=-0.5, height=0.2, axis=axis)
        self.plot_mu(axis=axis)
        plt.draw()
        plt.show(block=False)


class LRMean(LeftRightAgent):
    """Agent with three contexts in which only the mean of each contextg is
    inferred, while the SD is assumed to be fixed and is a free parameter of
    the model.

    """
    name = 'LRM'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mag_hypers = self.magnitudes
        self.mag_hypers[:, 1] = [0.2, 1, 1]
        self.mag_hypers_history = [self.mag_hypers]

    def update_magnitudes(self, ):
        r"""Updates the mean and sd hyperparameters of the mean of the
        magnitude according to the rule:
        \begin{align}
        \mu_{post} &= \frac{\sigma_{0}^2}{\sigma^2 + \sigma_0^2}x +
                        \frac{\sigma^2}{\sigma^2 + \sigma_0^2}\mu_{pre} \\
        \sd_{post}^2 &= \frac{\sigma^2\sigma_0^2}{\sigma^2 + \sigma_0^2}
        \end{align}
        Note that the SD of the mean of the magnitude remains fixed.

        """
        ix_context = self.sample_context(t=-2)  # todo: should be sampled from t-1
        mu_pre, sd_pre = self.mag_hypers[ix_context]
        mu_l, sd_l = self.predict_outcome()[1][ix_context]
        mu_l = mu_pre + self.hand_position - mu_l
        coeff = 1 / (sd_pre ** 2 + sd_l ** 2)
        mu_pos = sd_pre ** 2 * coeff * mu_l + \
            sd_l ** 2 * coeff * mu_pre
        sd_pos = np.sqrt(sd_l ** 2 * sd_pre ** 2 * coeff)
        # if ix_context == 1:
        #     ipdb.set_trace()
        mag_hypers = np.array(self.mag_hypers)
        mag_hypers[ix_context] = [mu_pos, sd_pos]
        magnitudes = np.array(self.magnitudes)
        magnitudes[ix_context][0] = mu_pos
        # magnitudes[ix_context][1] = mu_pos
        self.mag_hypers_history.append(mag_hypers)
        self.magnitude_history.append(magnitudes)
        self.magnitudes = magnitudes
        self.mag_hypers = mag_hypers


class LRMeanSD(LeftRightAgent):
    """Subclass that knows two contexts(plus baseline): one pointing left and one
    pointing right. The magnitude of these forces are inferred during the
    experiment.

    This model assumes that the environment generates the force field
    stochastically, sampling from a normal distribution with unknown parameters
    that are estimated after every observation. It uses NormalGamma priors for
    these parameters and drinks cold coffee. Psycho.

    """
    name = 'LRMS'

    def __init__(self, *args, **kwargs):
        """Initializes hyperparameters and calls LeftRightAgent.__init__ """
        super().__init__(*args, **kwargs)
        self.mag_hypers = [[angle, 1, sd / 0.01, 0.001]
                           for angle, sd in zip(self.angles, self.force_sds)]
        self.mag_hypers_history = [self.mag_hypers]

    def update_magnitudes(self):
        r"""Updates all four hyperparameters of the force magnitudes, using
        NormalGamma priors and posteriors.

        Note: for simplicity, a single value for the mean and the standard
        deviation is saved to self.magnitudes (and the rest of the code will
        treat these values as the mean and standard deviation of a
        Gaussian). The mean is simply the first hyperparameter and the standard
        deviation is $\beta / (\nu \alpha)$, where the hyperparameters are
        $\mu, \nu, \alpha, \beta$.

        """
        ix_context = self.sample_context()
        mu_l, sd_l = self.predict_outcome()[1][ix_context]
        mag_pars = self.mag_hypers[ix_context]
        mu_l = mag_pars[0] + self.hand_position - mu_l
        post_mag = [(mag_pars[0] * mag_pars[1] + mu_l) / (mag_pars[1] + 1),
                    mag_pars[1] + 1,
                    mag_pars[2] + 0.5,
                    mag_pars[3] + mag_pars[1] *
                    (mu_l - mag_pars[0]) ** 2 / 2 / (mag_pars[1] + 1)]
        mag_hypers = self.mag_hypers.copy()
        mag_hypers[ix_context] = post_mag
        self.mag_hypers = mag_hypers
        self.mag_hypers_history.append(mag_hypers)
        magnitudes = np.array(self.magnitudes)
        magnitudes[ix_context][0] = post_mag[0]
        magnitudes[ix_context][1] = post_mag[3] / post_mag[2]
        self.magnitudes = magnitudes
        self.magnitude_history.append(magnitudes)

    def sample_force(self, mode=None):
        """This function defines the logic to obtain a single number from
        the current estimation of the force. Implemented logics
        are 'mean', 'sample'.

        Parameters
        ----------
        mode : str
        One of {'mean', 'sample'}. If None, it will
        be taken from self.sample_context_mode.

        Returns
        -------
        sampled_context : int
        Integer to index the context.

        """
        if mode is None:
            mode = self.sample_force_mode
        ix_context = self.sample_context()
        force_pars = self.force_pars[ix_context]
        if mode == 'mean':
            sampled_force = force_pars[0]
        elif mode == 'sample':
            sampled_force = stats.norm(loc=force_pars[0],
                                       scale=force_pars[1]).rvs()
        return sampled_force

    def pandify_data(self):
        """Pandifies the history of the agent."""
        context = np.exp(self.log_context_history)
        max_context = np.argmax(context, axis=1)
        magnitudes = np.array(self.mag_hypers_history)
        mag_mu_mu = magnitudes[..., 0]
        mag_mu_sd = magnitudes[..., 1]
        mag_sd_mu = magnitudes[..., 2]
        mag_sd_sd = magnitudes[..., 3]
        hand = self.hand_position_history
        actions = self.action_history
        trial_number = np.arange(context.shape[0]) - 1
        aggregate = np.stack([*context.T, max_context,
                              *mag_mu_mu.T, *mag_mu_sd.T,
                              *mag_sd_mu.T, *mag_sd_sd.T, hand, actions,
                              trial_number],
                             axis=1)
        pandata = pd.DataFrame(aggregate,
                               columns=['con0', 'con1', 'con2',
                                        'con_t',
                                        'mag_mu_0', 'mag_mu_1', 'mag_mu_2',
                                        'mag_nu_0', 'mag_nu_1', 'mag_nu_2',
                                        'mag_alpha_0', 'mag_alpha_1',
                                        'mag_alpha_2',
                                        'mag_beta_0', 'mag_beta_1',
                                        'mag_beta_2',
                                        'hand', 'action', 'trial'])
        pandata.reset_index(drop=True, inplace=True)
        pandata.set_index('trial', inplace=True)
        return pandata
