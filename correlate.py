"""
Functions for generating intersubject correlation features.
"""
import os, sys
import logging
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from scipy.signal import medfilt

logger = logging.getLogger(__name__)


def pct_signal_change(ts):
    """Converts each timeseries (column of matrix ts) to % signal change."""
    means = np.tile(np.mean(ts, axis=1).T, [ts.shape[1], 1]).T
    return(((ts-means)/means) * 100)


def zscore(ts):
    """Converts each timeseries to have 0 mean and unit variance."""
    means = np.tile(np.mean(ts, axis=1).T, [ts.shape[1], 1]).T
    stdev = np.tile(np.std(ts, axis=1).T, [ts.shape[1], 1]).T
    return((ts-means)/stdev)


def tukeywin(win_length, alpha=0.75):
    """
    The Tukey window, also known as the tapered cosine window, is a cosine lobe
    of width alpha * N / 2 that is convolved with a rectangular window of width
    (1 - alpha / 2). At alpha = 1 it becomes rectangular, and at alpha = 0 it
    becomes a Hann window.

    http://leohart.wordpress.com/2006/01/29/hello-world/
    http://www.mathworks.com/access/helpdesk/help/toolbox/signal/tukeywin.html
    """
    # special cases
    if alpha <= 0:
        return np.ones(win_length)
    elif alpha >= 1:
        return np.hanning(win_length)

    # normal case
    x = np.linspace(0, 1, win_length)
    window = np.ones(x.shape)

    # first condition: 0 <= x < alpha/2
    c1 = x < alpha/2
    window[c1] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[c1] - alpha/2)))

    # second condition already taken care of
    # third condition 1 - alpha / 2 <= x <= 1
    c3 = x >= (1 - alpha/2)
    window[c3] = 0.5 * (1 + np.cos(2*np.pi/alpha * (x[c3] - 1 + alpha/2)))

    return window


def median_filter(ts, kernel_size=5):
    """
    Low-passes each time series using a n-d median filter. Useful in cases
    where one cannot assume Gaussian noise and/or would like to preserve
    edges within the data. We use this in fMRI to suppress the influence of
    outliers in the data. The default kernel size (5) is conservative, and does
    a nice job for typical fMRI run-lengths (120-200 TRs).
    """
    # init output array
    filtered_ts = np.zeros(ts.shape)

    # filter data per timeseries
    for i in np.arange(ts.shape[0]):
        filtered_ts[i, :] = medfilt(ts[i, :], kernel_size=int(kernel_size))

    return filtered_ts


def read_timeseries(db, row, col):
    """
    Return numpy array of the timeseries defined in the ith row, named column
    of input database db.
    """
    # in the single subject case
    if len(db.shape) == 1:
        timeseries_file = db[col]
    # in the multi subject case
    else:
        timeseries_file = db.iloc[row][col]
    logger.debug('reading timeseries file {}'.format(timeseries_file))
    try:
        return(np.genfromtxt(timeseries_file, delimiter=','))
    except:
        raise IOError('failed to parse timeseries {}'.format(timeseries_file))


def dynamic_connectivity(ts, win_length, win_step):
    """
    Calculates dynamic (sliding window) connectivity from input timeseries data,
    and outputs a roi x window matrix.
    """
    n_roi, n_tr = ts.shape

    # initialize the window
    idx_start = 0
    idx_end = win_length # no correction: 0 indexing balances numpy ranges

    # precompute the start and end of each window
    windows = []
    while idx_end <= n_tr-1:
        windows.append((idx_start, idx_end))
        idx_start += win_step
        idx_end += win_step

    # store the upper half of each connectivity matrix for each window
    idx_triu = np.triu_indices(n_roi, k=1)
    output = np.zeros((len(idx_triu[0]), len(windows)))

    # calculate taper (downweight early and late timepoints)
    taper = np.atleast_2d(tukeywin(win_length))

    for i, window in enumerate(windows):
        # extract sample, apply taper
        sample = ts[:, window[0]:window[1]] * np.repeat(taper, n_roi, axis=0)

        # keep upper triangle of correlation matrix
        test = np.corrcoef(sample)
        output[:, i] = np.corrcoef(sample)[idx_triu]

    return(output)


def calc_dynamic_connectivity(db, connectivity, win_length, win_step):
    """
    Calculates within-brain dynamic connnectivity of each participant in the
    submitted db. The connectivity features for all subjects are returned in a
    ROI x window matrix.
    """
    db_idx = db.index
    all_rs = []

    # loop through connectivity experiments
    for i, column in enumerate(connectivity):

        if len(db.shape) == 2:

            # loop through subjects
            for j, subj in enumerate(db_idx):

                try:
                    ts = read_timeseries(db, j, column)
                except IOError as e:
                    logger.error(e)
                    sys.exit(1)

                #ts = median_filter(ts)
                logger.debug('timeseries data: n_rois={}, n_timepoints={}'.format(ts.shape[0], ts.shape[1]))
                ts = zscore(ts)
                rs = dynamic_connectivity(ts, win_length, win_step)
                logger.debug('{} windows extracted'.format(rs.shape[1]))
                all_rs.append(rs)

        else:

            # only one subject, so don't loop
            try:
                ts = read_timeseries(db, 0, column)
            except IOError as e:
                logger.error(e)
                sys.exit(1)
            logger.debug('timeseries data: n_rois={}, n_timepoints={}'.format(ts.shape[0], ts.shape[1]))
            ts = zscore(ts)
            rs = dynamic_connectivity(ts, win_length, win_step)
            logger.debug('{} windows extracted'.format(rs.shape[1]))
            all_rs.append(rs)

    X = np.hstack(all_rs)
    logger.debug('correlation feature matrix shape: {}'.format(X.shape))

    return X


def calc_connectivity(db, connectivity):
    """
    Calculates within-brain connnectivity of each participant in the db. The
    connectivity features are concatenated for each participant and returned as
    the feature matrix X.
    """
    db_idx = db.index
    n = len(db)
    for i, column in enumerate(connectivity):

        # loop through subjects
        for j, subj in enumerate(db_idx):
            try:
                ts = read_timeseries(db, j, column)
            except IOError as e:
                logger.error(e)
                sys.exit(1)

            #ts = median_filter(ts)
            logger.debug('timeseries data: n_rois={}, n_timepoints={}'.format(ts.shape[0], ts.shape[1]))
            ts = zscore(ts)
            idx = np.triu_indices(ts.shape[0], k=1)
            rs = np.corrcoef(ts)[idx]

            # for the first timeseries, initialize the output array
            if j == 0:
                corrs = np.zeros((n, len(rs)))

            corrs[j, :] = rs

        # horizontally concatenate corrs into X (samples X features)
        if i == 0:
            X = corrs
        else:
            X = np.hstack((X, corrs))

    logger.debug('correlation feature matrix shape: {}'.format(X.shape))

    return X


def calc_xbrain(template_db, db, timeseries):
    """
    Calculates correlation of each participant in db with mean time series of
    everyone in the template db, excluding the participant's entry in the
    template if it exists. The features are concatenated for each participant
    and returned as the feature matrix X.
    """
    template_idx = template_db.index
    db_idx = db.index
    n = len(db)

    for i, column in enumerate(timeseries):

        # get a timepoint X roi X subject matrix from the template
        template_ts = get_column_ts(template_db, column)

        # loop through subjects
        for j, subj in enumerate(db_idx):
            if j == 0:
                n_roi, n_tr, n_subjects = template_ts.shape

                # for the first timeseries, initialize the output array
                # this is if we only store the diagonal
                #xcorrs = np.zeros((n, template_ts.shape[0]))

                # this is for storing the top half of the matrix
                idx_triu = np.triu_indices(n_roi)
                xcorrs = np.zeros((n, len(np.ravel(idx_triu))/2))

            try:
                ts = read_timeseries(db, j, column)
            except IOError as e:
                logger.error(e)
                sys.exit(1)

            logger.debug('timeseries data: n_rois={}, n_timepoints={}'.format(ts.shape[0], ts.shape[1]))
            ts = zscore(ts)

            # take the mean of the template, excluding this sample if shared
            unique_idx = template_idx != subj
            template_mean = np.mean(template_ts[:, :, unique_idx], axis=2)

            try:
                # diag of the intersubject corrs (upper right corner of matrix),
                # this includes only the correlations between homologous regions
                #rs = np.diag(np.corrcoef(ts, y=template_mean)[n_roi:, :n_roi])

                # full xbrain connectivity matrix
                rs = np.corrcoef(ts, y=template_mean)[n_roi:, :n_roi][idx_triu]
            except:
                raise Exception('xcorr dimension missmatch: subject {} dims={}, timeseries={}, template dims={}'.format(j, ts.shape, column, template_mean.shape))

            xcorrs[j, :] = rs

        # horizontally concatenate xcorrs into X (samples X features)
        if i == 0:
            X = xcorrs
        else:
            X = np.hstack((X, xcorrs))

    logger.debug('xbrain feature matrix shape: {}'.format(X.shape))

    return X


def get_column_ts(df, column):
    """
    Accepts a dataframe, and a timeseries column, reads the timeseries
    of all subjects, and returns a timepoint X roi X subject numpy array.
    """
    if type(column) == list:
        raise TypeError('column {} should be a valid pandas column identifier'.format(column))

    dims = read_timeseries(df, 0, column).shape
    n = len(df)
    template_ts = np.zeros((dims[0], dims[1], n))

    # collect timeseries
    for i in range(n):
        ts = read_timeseries(df, i, column)
        ts = zscore(ts)
        try:
            template_ts[:, :, i] = ts
        except:
            logger.error('{} timeseries file is the incorrect size (likely truncated timeseries)'.format(df.iloc[i][column]))
            sys.exit(1)

    return template_ts


def find_template(db, y, timeseries, group=-1):
    """
    Copies a subset of the input database into a template database.

    If group is defined (greater than -1), all of the subjects are used from
    that group to construct the template. Otherwise, all subjects are used in
    db.
    """
    if group > -1:
        return(db.loc[db[y] == group])
    else:
        return(db)


