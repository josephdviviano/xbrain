"""
Functions for generating intersubject correlation features.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Force matplotlib to not use any Xwindows backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def read_timeseries(db, row, col):
    """
    Return numpy array of the timeseries defined in the ith row, named column
    of input database db.
    """
    timeseries_file = db.iloc[row][col]
    try:
        return(np.genfromtxt(timeseries_file, delimiter=','))
    except:
        raise IOError('failed to parse timeseries {}'.format(timeseries_file))


def pct_signal_change(ts):
    """Converts each timeseries (column of matrix ts) to % signal change."""
    means = np.tile(np.mean(ts, axis=0), [ts.shape[0], 1])
    return(((ts-means)/means) * 100)


def calc_xbrain(template_ts, participant_pop, timeseries):
    """
    Calculates correlation of each participant in the dataframe participant_pop
    with the appropriate slice of the input template matrix. The features are
    concatenated for each participant and returned as the feature matrix X.
    """
    n = len(participant_pop)
    X = np.zeros((n, template_ts.shape[1]*len(timeseries)))
    logger.debug('xbrain feature matrix shape: {}'.format(X.shape))

    for i in range(n):
        xcorrs = np.array([])
        for j, column in enumerate(timeseries):

            try:
                ts = read_timeseries(participant_pop, i, column)
            except IOError as e:
                logger.error(e)
                sys.exit(1)

            ts = pct_signal_change(ts)
            # diag of the cross-variable corrs (upper right corner of matrix)
            try:
                rs = np.diag(np.corrcoef(ts.T, y=template_ts[:, :, j].T)[ts.shape[1]:, :ts.shape[1]])
                xcorrs = np.concatenate((xcorrs, rs))
            except:
                raise Exception('xcorr failed due to missmatched dimensions: subject {} dims={}, template {} dims={}'.format(i, ts.shape, j, template_ts[:,:,j].shape))
        X[i, :] = xcorrs

    # remove values less than zero (not meaningful)
    X[X < 0] = 0

    return X


def get_template_ts(template, timeseries):
    """
    Accepts a template dataframe, and for each timeseries column, takes the
    average timeseries activity. Returns a roi X timepoint X timeseries numpy
    matrix.
    """
    dims = read_timeseries(template, 0, timeseries[0]).shape
    template_ts = np.zeros((dims[0], dims[1], len(timeseries)))
    n = len(template)

    # take mean timeseries across template population for each timeseries type
    for i, column in enumerate(timeseries):
        for j in range(n):
            ts = np.genfromtxt(template[column].iloc[j], delimiter=',')
            ts = pct_signal_change(ts)
            template_ts[:, :, i] += ts
        template_ts[:, :, i] = template_ts[:, :, i] / float(n)

    return template_ts


def find_template(db, n_template, y, timeseries, percentile=50, group=-1):
    """
    Copies a subset of the input database into a template database.

    Can be run in two modes: 'group' and 'percentile'.

    If group is defined (greater than -1), a random number (n_template) of
    subjects are used from that group to construct the template. If n_template
    is greater than the number of subjects available in the group, this will
    use all of them.

    If group is not defined, this will take the defined percentile of the input
    variable y. This will take n_subjects closest to the defined percentile. If
    n_subjects is larger than the number of available subjects in the database,
    this will use all of them.
    """
    if group > -1:

        # subset database to the group of interest
        template = db.loc[db[y] == group]

        # shuffle rows
        template = template.iloc[np.random.permutation(len(template))]

        # if we get a valid n_template input, take a subset of the subjects
        if n_template < len(template):
            template = template.iloc[:n_template]

    else:
        db = db.sort(y) # sort database so similar y values are in adjacent rows
        db = db.reset_index(drop=True) # renumber rows for easy indexing

        # find the target value without taking midpoints (favor higher value)
        target = np.percentile(db[y], percentile, interpolation='higher')
        target_idx = np.percentile(db.loc[db[y] == target].index, 50, interpolation='lower')
        logger.debug('target percentile={}, score={}'.format(percentile, target))

        # get template indicies
        idx_lo = target_idx - np.floor(n_template/2.0)
        idx_hi = target_idx + np.ceil(n_template/2.0)

        # remove impossible indicies
        if idx_hi >= len(db):
            idx_hi = len(db)-1 # correct for 0-based indexing
        if idx_lo < 0:
            idx_lo = 0

        # take subset of the subjects (or use the whole test set)
        template_idx = np.arange(idx_lo, idx_hi+1)
        logger.debug('template subjects: {} - {}'.format(int(idx_lo), int(idx_hi)))
        template = db.loc[template_idx]

    # get mean template timeseries
    ts = get_template_ts(template, timeseries)

    return ts


def plot_features(X, path, i, X_low=None):
    """
    Plots the cross brain correlation features calculated. In the two-template
    case, plots each X matrix individually, and a 3rd matrix of their
    difference. Negative correlations are nonsense for cross brain correlations,
    and are set to 0 for simplicity. Plots are written out to the directory
    specified as path.
    """
    if not os.path.isdir(path):
        raise Exception('path {} is not a directory'.format(path))

    if X_low is not None:
        plt.subplot(311)
        plt.imshow(X, vmin=0, vmax=0.5, cmap=plt.cm.Reds)
        plt.title('X high')
        plt.subplot(312)
        plt.imshow(X_low, vmin=0, vmax=0.5, cmap=plt.cm.Reds)
        plt.title('X low')
        plt.subplot(313)
        plt.imshow(X - X_low, vmin=-0.5, vmax=0.5, cmap=plt.cm.RdBu_r)
        plt.title('X ratio')
        plt.savefig(os.path.join(path, 'X_iter={}.pdf'.format(i+1)))
        plt.close()

    else:
        plt.imshow(X, vmin=-0, vmax=0.5, cmap=plt.cm.Reds)
        plt.colorbar()
        plt.savefig(os.path.join(path, 'X_iter={}.pdf'.format(i+1)))
        plt.close()


