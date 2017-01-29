"""Helper functions and data munging."""

import pickle
import numpy as np
import pandas as pd
import logging
from copy import copy

logger = logging.getLogger(__name__)

def reorder(X, idx, symm=False):
    """
    Reorders the rows of a matrix. If symm is True, this simultaneously reorders
    the columns and rows of a matrix by the given index.
    """
    if symm:
        assert_square(X)
        X = X[:, idx]
    X = X[idx, :]

    return(X)


def assert_square(X):
    if len(X.shape) != 2 or X.shape[0] != X.shape[1]:
        raise Exception("Input matrix must be square")


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


def find_template(db, n_template, predict, percentile=50):
    """
    Copies a subset of the input database into a template database, which is n
    subjects of the input population with values closest to the target y score
    of interest (by percentile, default is the median, or 50%). Always picks an
    actual sample for the target percentile, therefore, if an even number of
    values is submitted, will take the value greater than the midpoint value
    normally calculated for the median.
    """
    db = db.sort(predict)

    # find the target value without taking midpoints (favor higher value)
    target = np.percentile(db[predict], percentile, interpolation='higher')
    target_idx = np.percentile(np.where(db[predict] == target)[0], 50, interpolation='lower')
    logger.debug('target percentile={}, score={}'.format(percentile, target))

    # get template indicies
    if is_even(n_template):
        idx_lo = target_idx - n_template/2
        idx_hi = target_idx + n_template/2
    else:
        idx_lo = target_idx - np.floor(n_template/2.0)
        idx_hi = target_idx + np.ceil(n_template/2.0)

    # split database into template and participant samples
    template_idx = np.arange(idx_lo, idx_hi+1)
    logger.debug('template subjects: {} - {}'.format(int(idx_lo), int(idx_hi)))
    template = db.iloc[template_idx]

    return template


def is_probability(x):
    """True if x is a float between 0 and 1, otherwise false."""
    if x > 0 and x < 1:
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


def gather_dv(db, columns):
    """
    Returns a numpy vector of the predicted column. Cutoff is a percentage
    (0 < p < 0.5). If cutoff specified, returns a binary vector (0 = lower than
    cutoff, 1 = higher than cutoff). The maximum cutoff is 50%, or the median
    of the sample.
    """
    for i, col in enumerate(columns):
        tmp = np.array(db[col])
        if i == 0:
            y = tmp
        else:
            y = np.vstack((y, tmp))

    return(y)


def make_dv_groups(y, cutoff):
    """
    Accepts a numpy vector of the dependent variable y. All scores lower than
    the submitted percentile cutoff are set to 0, and the rest are set to 1.
    Used to turn continuous variables into groups for outlier detection.
    """
    cutoff = np.percentile(y, cutoff*100)
    idx_lo = np.where(y < cutoff)[0]
    idx_hi = np.where(y >= cutoff)[0]
    y[idx_lo] = 0
    y[idx_hi] = 1

    return y


def pickle_it(my_data, save_path):
    f = open(save_path, 'wb')
    pickle.dump(my_data, f)
    f.close()


