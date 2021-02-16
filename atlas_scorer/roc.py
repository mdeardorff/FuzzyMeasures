"""This module handles the ROC class used by the scorer."""

import copy
import logging

import matplotlib.pyplot as plt
import numpy as np

from atlas_scorer.errors import AtlasScorerError


def atlas_score_roc(x, y):
    """
    Generate and ROC object from confidence values (x) and binary labels (y).
    
    :param Iterable x: Confidence values.
    :param Iterable y: Labels containing 0 or 1
    :return: An AtlasMetricROC object.
    :rtype: AtlasMetricROC
    """

    x = np.array(x)
    y = np.array(y)

    unique_y = np.unique(y)

    if x.shape[0] == 0 and y.shape[0] == 0:
        return AtlasMetricROC([], [], [], [], 0, 0)

    if len(unique_y) != 2:
        if (len(unique_y) == 1) and (unique_y[0] == 0 or unique_y[0] == 1):
            log = logging.getLogger(__name__)
            log.warning('Only one class of data was provided; this will result '
                        'in NaN PD or PFA values.')
            unique_y = [0, 1]
        else:
            raise AtlasScorerError('Two unique classes required.')

    if len(x) != len(y):
        raise AtlasScorerError('x and y must be equal length.')

    if set(unique_y) != {0, 1}:
        y[y == unique_y[0]] = 0
        y[y == unique_y[1]] = 1

    sort_idx = np.argsort(x)[::-1]
    sorted_ds = x[sort_idx]
    sorted_y = y[sort_idx]

    nan_spots = np.isnan(sorted_ds)

    # Number of detections as a function of threshold.
    prob_det = np.copy(sorted_y)

    # Number of false alarms as a function of threshold.
    prob_fa = 1 - prob_det

    # Detect and handle ties
    if len(sorted_ds) > 1:
        is_tied_with_next = np.concatenate(
            (sorted_ds[0:-1] == sorted_ds[1:], [False]))
    else:
        is_tied_with_next = [False]

    # If there are any ties we need to figure out the tied regions and set each
    # of the ranks to the average of the tied ranks.
    if any(is_tied_with_next):
        # use prepend=0 to ensure that we aggregate ties even when the
        # first entry is tied (the diff will generate a "1" in
        # diff_is_tied_with_next[0])
        diff_is_tied_with_next = np.diff(is_tied_with_next.astype(np.int),
                                         prepend=0)

        # Start and stop regions of the ties.
        idx1 = np.flatnonzero(diff_is_tied_with_next == 1)
        idx2 = np.flatnonzero(diff_is_tied_with_next == -1) + 1

        # For each tied region we set the first value of PD (or PFA) in the tied
        # region equal to the number of hits (or non-hits) in the range and we
        # set the rest to zero.  This makes sure that when we cumsum (integrate)
        # we get all of the tied values at the same time.
        for s, e in zip(idx1, idx2):
            prob_det[s] = np.sum(prob_det[s:e])
            prob_det[s+1:e] = 0

            prob_fa[s] = np.sum(prob_fa[s:e])
            prob_fa[s+1:e] = 0

    nh1 = sorted_y.sum()
    nh0 = len(sorted_y) - nh1

    # NaNs are not counted as detections or false alarms.
    prob_det[nan_spots & (sorted_y == 1)] = 0
    prob_fa[nan_spots & (sorted_y == 0)] = 0

    num_fa = np.cumsum(prob_fa)
    prob_det = np.cumsum(prob_det) / nh1
    prob_fa = num_fa / nh0

    prob_det = np.concatenate(([0], prob_det))
    prob_fa = np.concatenate(([0], prob_fa))
    num_fa = np.concatenate(([0], num_fa))
    thresholds = np.concatenate(([np.inf], sorted_ds))

    return AtlasMetricROC(prob_fa, prob_det, num_fa, thresholds, nh1, nh0)


class AtlasMetricROC:
    """ROC object for handling metadata associated with an ROC."""
    def __init__(self, pf, pd, nfa, tau, num_targets, num_non_targets):
        self.pf = pf
        self.pd = pd
        self.nfa = nfa
        self.tau = tau
        self.nTargets = num_targets
        self.nNonTargets = num_non_targets

        self.farDenominator = None

    @property
    def far(self):
        """Far is a dependent property which is just nfa/farDenominator."""
        if self.farDenominator is None:
            return None
        else:
            return self.nfa / self.farDenominator

    @property
    def tp(self):
        """True Positives"""
        return self.pd * self.nTargets

    @property
    def fp(self):
        """False Positives"""
        return self.pf * self.nNonTargets

    @property
    def tn(self):
        """True Negatives"""
        return (1 - self.pf) * self.nNonTargets

    @property
    def fn(self):
        """False Negatives"""
        return (1 - self.pd) * self.nTargets

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self):
        return self.pd

    @property
    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    @property
    def f1(self):
        return (2 * self.tp) / (2 * self.tp + self.fp + self.fn)

    @property
    def auc(self):
        return np.trapz(self.pd, self.pf)

    def pd_at_far_vals(self, far_points):

        far_points = np.array(far_points, dtype=float)
        if self.farDenominator is None:
            raise AtlasScorerError(
                'Cannot calculate FAR values since `farDenominator` is not set')
        tmp_far = np.concatenate((self.far, [np.inf]))
        tmp_pd = np.concatenate((self.pd, self.pd[-1:]))  # Change from PRT

        ind_fn = np.vectorize(lambda far_val: np.nonzero(tmp_far > far_val)[0][0])

        inds = ind_fn(far_points)
        return tmp_pd[inds]

    def far_at_pd_vals(self, pd_points):
        """
        Calculate FAR values corresponding to the provided pd_points

        Args:
            pd_points(float or np.ndarray): Point(s) at which to calculate the
                corresponding FAR values

        Returns:
            (np.ndarray of np.float): FAR values corresponding to `pd_points`
        """
        pd_points = np.array(pd_points, dtype=float)
        if self.farDenominator is None:
            raise AtlasScorerError(
                'Cannot calculate FAR values since `farDenominator` is not set')
        tmp_far = np.concatenate((self.far, [np.inf]))
        tmp_pd = np.concatenate((self.pd, [np.inf]))

        ind_fn = np.vectorize(lambda pd_val: np.nonzero(tmp_pd >= pd_val)[0][0])

        inds = ind_fn(pd_points)
        return tmp_far[inds]

    def pauc_pf(self, max_pf):
        """
        Calculate partial-AUC under the PD vs PF curve up to `max_pf`

        Args:
            max_pf(float): Max PF value for partial-AUC calculation

        Returns:
            (float): The partial AUC for the PD vs PF ROC curve
        """
        pd, pf = self.pd, self.pf

        pf_inds = pf <= max_pf
        pd = pd[pf_inds]
        pf = pf[pf_inds]

        pd = np.concatenate((pd, pd[-1:]))
        pf = np.concatenate((pf, [max_pf]))

        return np.trapz(pd, pf)

    def pauc_far(self, max_far):
        """
        Calculate partial-AUC under the PD vs FAR curve up to `max_far`

        Args:
            max_far(float): Max FAR value for partial-AUC calculation

        Returns:
            (float): The partial AUC for the PD vs FAR ROC curve
        """
        pd, far = self.pd, self.far

        far_inds = far <= max_far
        pd = pd[far_inds]
        far = far[far_inds]

        pd = np.concatenate((pd, pd[-1:]))
        far = np.concatenate((far, [max_far]))

        return np.trapz(pd, far)

        # max_far = max_far if max_far is not None else np.inf  # TODO: Bug in PRT?

        # n_tgt = self.nTargets
        # u_pd = np.linspace(0, 1, n_tgt + 1)
        # u_pd_far = self.far_at_pd_vals(u_pd)
        #
        # c_x = u_pd_far.flatten()
        # c_y = u_pd.flatten()
        # keep = c_x <= max_far
        # c_x = c_x[keep]
        # c_y = c_y[keep]
        #
        # # Append a last point @ maxFar to ensure we get the final
        # # rectangle up to the requested FAR - note: Can optionally
        # # include the trapezoidal extension... but that's not
        # # actually the PD you would get based on the data we've
        # # seen.  It's a conundrum and the extension is complicated
        # c_x = np.concatenate((c_x, [max_far]))
        # c_y = np.concatenate((c_y, c_y[-1:]))
        #
        # return np.trapz(c_y, c_x)

    def copy(self):
        """Return deep-copy of this `AtlasMetricROC` instance"""
        return copy.deepcopy(self)

    def write_csv(self, csv_file):
        """Write ROC object to file."""
        v = np.column_stack((self.nfa, self.far, self.pd, self.tau))
        np.savetxt(csv_file, v, fmt='%.8f',
                   header="nfa, far, pd, tau", delimiter=',', comments='')

    def _plot_xy(self, x, y, ax=None, title=None, xlabel=None, ylabel=None,
                 label=None, xlim_args=None, plot_kwargs=None):
        """Common plotting code to make standard plots"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        if xlim_args is None:
            xlim_args = {}
        if plot_kwargs is None:
            plot_kwargs = {}
        ax.plot(x, y, label=label, **plot_kwargs)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        ax.grid(which='both', alpha=0.4)
        if label is not None:
            ax.legend(loc='lower right')
        ax.autoscale(True)

        # Add small 0.4% padding on either side for x-axis so lines at x=0 or 1
        # display correctly (typically get occluded if xlim is [0, 1])
        x_lims = (min(x), max(x))
        x_lims_padding = (x_lims[1] - x_lims[0]) * 0.004
        x_lims_left = x_lims[0] - x_lims_padding
        x_lims_right = x_lims[1] + x_lims_padding

        x_lims_dict = {**{'left': x_lims_left, 'right': x_lims_right},
                       **xlim_args}
        ax.set_xlim(**x_lims_dict)

        ax.set_ylim(bottom=0, top=1.005)
        if title is not None:
            ax.set_title(title)
        return ax

    def plot_roc(self, ax=None, title='', xlabel='$P_{Fa}$', ylabel='$P_D$',
                 label=None, xlim_args=None, plot_kwargs=None):
        xlim_args = {} if xlim_args is None else xlim_args
        return self._plot_xy(self.pf, self.pd, ax=ax, title=title,
                             xlabel=xlabel, ylabel=ylabel, label=label,
                             xlim_args={**{'right': 1}, **xlim_args},
                             plot_kwargs=plot_kwargs)

    def plot_far(self, ax=None, title='', xlabel='$FAR$', ylabel='$P_D$',
                 label=None, xlim_args=None, plot_kwargs=None):
        if self.far is None:
            raise AttributeError(
                'FAR is `None`. Ensure that `farDenominator` has been '
                'set correctly')
        return self._plot_xy(self.far, self.pd, ax=ax, title=title,
                             xlabel=xlabel, ylabel=ylabel, label=label,
                             xlim_args=xlim_args, plot_kwargs=plot_kwargs)

    def plot_semilog_far(self, ax=None, title='', xlabel='$FAR$', ylabel='$P_D$',
                 label=None, min_far=1e-5, plot_kwargs=None):
        ax = self.plot_far(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel,
                           label=label, xlim_args={'left': min_far},
                           plot_kwargs=plot_kwargs)
        ax.set_xscale('log')
        if label is not None:
            ax.legend(loc='best')
        return ax

    def plot_prec_recall(self, ax=None, title='', xlabel='$Recall$',
                         ylabel='$Precision$', label=None, plot_kwargs=None):
        return self._plot_xy(self.recall, self.precision, ax=ax, title=title,
                             xlabel=xlabel, ylabel=ylabel, label=label,
                             xlim_args={'right': 1}, plot_kwargs=plot_kwargs)
