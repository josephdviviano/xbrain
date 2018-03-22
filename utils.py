"""Helper functions and data munging."""

import pickle
import numpy as np
import pandas as pd
import nibabel as nib
import os, sys
import logging
from copy import copy

logger = logging.getLogger(__name__)

def assert_columns(db, columns):
    if not is_column(db, columns):
        logger.error('not all columns {} found'.format(columns))
        sys.exit(1)


def assert_square(X):
    if len(X.shape) != 2 or X.shape[0] != X.shape[1]:
        raise Exception("Input matrix must be square")


def is_probability(x):
    """True if x is a float between 0 and 1, otherwise false."""
    if x >= 0 and x <= 1:
        return True
    return False


def is_column(df, column):
    """
    True if column is in pandas dataframe df. If column is a list, checks all of
    them.
    """
    if type(column) == str:
        if column in df.columns:
            return True
        return False

    elif type(column) == list:
        for c in column:
            if not is_column(df, c):
                return False
        return True


def is_even(n):
    """True if n is even, else false."""
    if n % 2 == 0:
        return True
    return False


def split_columns(variable):
    """
    Splits the input variable, which is either None or a comma delimited string.
    """
    if variable:
        return(variable.split(','))
    return(variable)


def clean(X):
    """
    Replaces nan and inf values in numpy array with zero. If any columns are all
    0, removes them completely.
    """
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0
    logger.debug('X matrix has {} bad values (replaced with 0)'.format(np.sum(X == 0)))

    idx_zero = np.where(np.sum(np.abs(X), axis=0) == 0)[0] # find all zero cols

    if len(idx_zero) > 0:
        logger.debug('removing {} columns in X that are all 0'.format(len(idx_zero)))
        idx = np.arange(X.shape[1])
        idx = np.setdiff1d(idx, idx_zero)
        X = X[:, idx]

    return(X)


def reorder(X, idx, symm=False):
    """
    Reorders the rows of a matrix. If symm is True, this simultaneously reorders
    the columns and rows of a matrix by the given index.
    """
    if symm:
        assert_square(X)
        X = X[:, idx]

    if X.shape[0] != len(idx):
        logger.warn('reorg IDX length {} does not match the rows of X {}'.format(len(idx), X.shape[0]))

    X = X[idx, :]

    return(X)


def full_rank(X):
    """Ensures input matrix X is not rank deficient."""
    if len(X.shape) == 1:
        return True

    k = X.shape[1]
    rank = np.linalg.matrix_rank(X)
    if rank < k:
        return False

    return True


def scrub_data(x):
    """
    Removes NaNs from a vector, and raises an exception if the data is empty.
    """
    x[np.isnan(x)] = 0
    if np.sum(x) == 0:
        raise Exception('vector contains no information')

    return x


def gather_columns(db, columns):
    """
    Returns a numpy matrix of the specified columns.
    """
    for i, col in enumerate(columns):
        tmp = np.array(db[col])
        if i == 0:
            X = tmp
        else:
            X = np.vstack((X, tmp))

    return(X.T)


def read_statmap(db, row, col, roi_mask):
    """
    Return numpy array of the timeseries defined in the ith row, named column
    of input database db.
    """
    # in the single subject case
    if len(db.shape) == 1:
        statmap_file = db[col]
    # in the multi subject case
    else:
        statmap_file = db.iloc[row][col]

    logger.debug('reading statmap file {}'.format(statmap_file))
    statmap = nib.load(statmap_file).get_data()
    roi_file = nib.load(roi_mask).get_data()

    try:
        assert(statmap.shape == roi_file.shape)
    except AssertionError:
        logger.error('input statmap dims {} does not match ROI mask {}'.format(statmap.shape, roi_file.shape))
        sys.exit(1)

    # take all non-zero voxels as an ROI, init output x
    rois = np.unique(roi_file)
    rois = rois[rois > 0]
    x = np.zeros(len(rois))

    # take mean within each ROI
    for i, roi in enumerate(rois):
        idx = np.where(roi_file == roi)
        x[i] = np.mean(statmap[idx])

    return(x)


def gather_stats(db, statmaps, roi_mask):
    """
    Calculates mean stat within each roi of the supplied mask of each
    participant in the db and returned as the feature matrix X.
    """
    db_idx = db.index
    n = len(db)
    for i, column in enumerate(statmaps):

        # loop through subjects
        for j, subj in enumerate(db_idx):
            try:
                stat = read_statmap(db, j, column, roi_mask)
            except IOError as e:
                logger.error(e)
                sys.exit(1)

            logger.debug('stat data: n_rois={}'.format(stat.shape[0]))

            # for the first timeseries, initialize the output array
            if j == 0:
                stats = np.zeros((n, stat.shape[0]))

            stats[j, :] = stat

        # horizontally concatenate corrs into X (samples X features)
        if i == 0:
            X = stats
        else:
            X = np.hstack((X, stats))

    logger.debug('stats feature matrix shape: {}'.format(X.shape))

    return X



def make_dv_groups(y, cutoff):
    """
    Accepts a numpy vector of the dependent variable y. All scores lower than
    the submitted percentile cutoff are set to 0, and the rest are set to 1.
    Used to turn continuous variables into groups for outlier detection.
    """
    logger.info('partitioning y at the {}th percentile'.format(cutoff*100))
    cutoff = np.percentile(y, cutoff*100)
    idx_lo = np.where(y < cutoff)[0]
    idx_hi = np.where(y >= cutoff)[0]
    y[idx_lo] = 0
    y[idx_hi] = 1

    return y


def load_biotype(biotype_mdl):
    """loads a saved biotype model"""
    if not os.path.isfile(biotype_mdl):
        raise Exception('{} does not exist'.format(biotype_mdl))

    mdl = np.load(biotype_mdl)

    # expected in a valid biotype model (see stats.estimate_biotypes)
    keys = ['reg', 'n_cc', 'n_clst', 'clusters', 'cancorrs', 'centroids',
        'comps_X', 'comps_y', 'ws_X', 'ws_y', 'idx_X', 'X', 'y', 'y_names', 'cca']

    for key in keys:
        if key not in mdl.keys():
            raise Exception('biotype model does not contain variable {}'.format(key))

    return(mdl)

def pickle_it(my_data, save_path):
    f = open(save_path, 'wb')
    pickle.dump(my_data, f)
    f.close()


