"""Helper functions and data munging."""

import pickle
import numpy as np
import pandas as pd
import logging

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


def split_samples(db, n_template, predict, percentile=50):
    """
    Splits input database into a template database, which is n subjects of the
    input population with values closest to the target percentile prediction
    score of interest (default is the median, or 50%), and the remainer of the
    sample population. Always picks an actual sample for the target percentile,
    therefore, if an even number of values is submitted, will take the value
    greater than the midpoint value normally calculated for the median.
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
    participants_idx = np.setdiff1d(np.arange(db.shape[0]), template_idx)
    logger.debug('template subjects: {} - {}'.format(idx_lo, idx_hi))
    logger.debug('participants: {}'.format(participants_idx))
    template = db.iloc[template_idx]
    participants = db.iloc[participants_idx]

    return template, participants


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


def gather_dv(participant_pop, predict):
    """Returns a numpy vector of the predicted column."""
    return(np.array(participant_pop[predict]))


def pickle_it(my_data, save_path):
    f = open(save_path, 'wb')
    pickle.dump(my_data, f)
    f.close()


