#!/usr/bin/env python
"""Routines for relating neural activity with clinical variables."""

import os, sys, glob, copy
import collections
import logging
import random
import string

import numpy as np
import scipy as sp
import scipy.cluster.hierarchy as sch
from scipy import stats
from scipy import linalg
from scipy.stats import mode
import pandas as pd
from sklearn import preprocessing
from sklearn import grid_search
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as mse

import matplotlib
matplotlib.use('Agg')   # Force matplotlib to not use any Xwindows backend
import matplotlib.pyplot as plt
import seaborn as sns

import xbrain.utils as utils

logger = logging.getLogger(__name__)

def r_to_z(R):
    """Fischer's r-to-z transform on a matrix (elementwise)."""
    return(0.5 * np.log((1+R)/(1-R)))


def r_to_d(R):
    """Converts a correlation matrix R to a distance matrix D."""
    return(np.sqrt(2*(1-R)))


def standardize(X):
    """z-scores each column of X."""
    return((X - X.mean(axis=0)) / X.std(axis=0))


def sig_cutoffs(null, two_sided=True):
    """Returns the significance cutoffs of the submitted null distribution."""
    if two_sided:
        sig = np.array([np.percentile(null, 2.5), np.percentile(null, 97.5)])
    else:
        sig = np.array([np.percentile(null, 5), np.percentile(null, 95)])

    return(sig)


def gowers_matrix(D):
    """Calculates Gower's centered matrix from a distance matrix."""
    utils.assert_square(D)

    n = float(D.shape[0])
    o = np.ones((n, 1))
    I = np.identity(n) - (1/n)*o.dot(o.T)
    A = -0.5*(np.square(D))
    G = I.dot(A).dot(I)

    return(G)


def hat_matrix(X):
    """
    Caluclates distance-based hat matrix for an NxM matrix of M predictors from
    N variables. Adds the intercept term for you.
    """
    X = np.hstack((np.ones((X.shape[0], 1)), X)) # add intercept
    Q1, R1 = np.linalg.qr(X)
    H = Q1.dot(Q1.T)

    return(H)


def calc_F(H, G, m=None):
    """
    Calculate the F statistic when comparing two matricies.
    """
    utils.assert_square(H)
    utils.assert_square(G)

    n = H.shape[0]
    I = np.identity(n)
    IG = I-G

    if m:
        F = (np.trace(H.dot(G).dot(H)) / (m-1)) / (np.trace(IG.dot(G).dot(IG)) / (n-m))
    else:
        F = (np.trace(H.dot(G).dot(H))) / np.trace(IG.dot(G).dot(IG))

    return F


def permute(H, G, n=10000):
    """
    Calculates a null F distribution from a symmetrically-permuted G (Gower's
    matrix), from the between subject connectivity distance matrix D, and a the
    H (hat matrix), from the original behavioural measure matrix X.

    The permutation test is accomplished by simultaneously permuting the rows
    and columns of G and recalculating F. We do not need to account for degrees
    of freedom when calculating F.
    """
    F_null = np.zeros(n)
    idx = np.arange(G.shape[0]) # generate our starting indicies

    for i in range(n):
        idx = np.random.permutation(idx)
        G_perm = utils.reorder(G, idx, symm=True)
        F_null[i] = calc_F(H, G_perm)

    F_null.sort()

    return F_null


def variance_explained(H, G):
    """
    Calculates variance explained in the distance matrix by the M predictor
    variables in X.
    """
    utils.assert_square(H)
    utils.assert_square(G)

    return((np.trace(H.dot(G).dot(H))) / np.trace(G))


def mdmr(X, Y):
    """
    Multvariate regression analysis of distance matricies: regresses variables
    of interest X (behavioural) onto a matrix representing the similarity of
    connectivity profiles Y.

    Zapala & Schork, 2006. Multivariate regression analysis of distance matrices
    for testing association between gene expression patterns related variables.
    PNAS 103(51)
    """
    if not utils.full_rank(X):
        raise Exception('X is not full rank:\ndimensions = {}'.format(X.shape))

    X = standardize(X)   # mean center and Z-score all cognitive variables
    R = np.corrcoef(Y)   # correlation distance between each cross-brain correlation vector
    D = r_to_d(R)        # distance matrix of correlation matrix
    G = gowers_matrix(D) # centered distance matrix (connectivity similarities)
    H = hat_matrix(X)    # hat matrix of regressors (cognitive variables)
    F = calc_F(H, G)     # F test of relationship between regressors and distance matrix
    F_null = permute(H, G)
    v = variance_explained(H, G)

    return F, F_null, v


def backwards_selection(X, Y):
    """
    Performs backwards variable selection on the input data.
    """

    return False


def individual_importances(X, Y):
    """
    Runs MDMR individually for each variable. If the variable is deemed
    significant, the variance explained is recorded, otherwise it is reported
    as 0. Returns a vector of variance explained.
    """
    m = X.shape[1]
    V = np.zeros(m)
    for test in range(m):
        X_test = np.atleast_2d(X[:, test]).T # enforces a column vector
        F, F_null, v = mdmr(X_test, Y)
        thresholds = sig_cutoffs(F_null, two_sided=False)
        if F > thresholds[1]:
            V[test] = v
        else:
            V[test] = 0
        print('tested variable {}/{}'.format(test+1, m))

    return V


def pca_reduce(X, n=1):
    """Uses PCA to return the top n components of the data as a matrix."""

    mean = np.mean(X, axis=0)

    # calculate the covariance matrix from centered data
    X = X - np.mean(X)
    R = np.cov(X, rowvar=True)

    # calculate eigenvectors & eigenvalues of the covariance matrix
    evals, evecs = linalg.eigh(R)

    # sort by explained variance
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]

    # reduce to n components
    evecs = evecs[:, :n]
    recon = np.dot(evecs.T, X)

    # sign flip to match mean if only taking one component
    if n == 1:
        recon = recon.flatten()
        corr = np.corrcoef(np.vstack((recon, mean)))[0,1]
        if corr < 0:
            recon = recon * -1

    return(recon)


def cv_loop(model_clf, hyperparams, X_train, y_train):
    """
    Uses cross validation to do a grid search on the hyperparameter dictionary
    input.
    """
    clf = grid_search.GridSearchCV(model_clf, hyperparams, cv=3, verbose=0)
    clf.fit(X_train, y_train)

    return clf


def make_classes(y):
    """transforms label values for classification"""
    le = preprocessing.LabelEncoder()
    le.fit(y)
    return(le.transform(y))


def classify(X_train, X_test, y_train, y_test, model='RFC'):
    """
    Trains the selected classifier once on the submitted training data, and
    compares the predicted outputs of the test data with the real labels.
    Includes a hyper-parameter cross validation loop, the 'innermost' loop.
    Returns a set of metrics collected from the hyperparameter grid search and
    the test error.
    """
    if X_train.shape[0] != y_train.shape[0]:
        raise Exception('X_train shape {} does not equal y_train shape {}'.format(X_train.shape[0], y_train.shape[0]))
    if X_test.shape[0] != y_test.shape[0]:
        raise Exception('X_test shape {} does not equal y_test shape {}'.format(X_test.shape[0], y_test.shape[0]))

    hp_dict = collections.defaultdict(list)
    hp_mode = {}
    r_train, r_test, R2_train, R2_test, MSE_train, MSE_test, pred_scores, real_scores = [], [], [], [], [], [], [], []

    # for testing various models, includes grid search settings
    if model == 'LR_L1':
        model_clf = Lasso()
        hyperparams = {'alpha':[0.2, 0.1, 0.05, 0.01]}
        scale_data = True
        feat_imp = True
        continuous = True
    elif model == 'SVR':
        model_clf = SVR()
        hyperparams = {'kernel':['linear','rbf'], 'C':[1,10,25]}
        scale_data = True
        feat_imp = True
        continuous = True
    elif model == 'RFR':
        model_clf = RandomForestRegressor(n_jobs=6)
        hyperparams = {'n_estimators':[10,50,100,200,500], 'min_samples_split':[2,5,10,20,50]}
        scale_data = False
        feat_imp = True
        continuous = True
    elif model == 'RFC':
        model_clf = RandomForestClassifier(n_jobs=6)
        hyperparams = {'n_estimators':[10,50,100,200,500], 'min_samples_split':[2,5,10,20,50]}
        scale_data = False
        feat_imp = True
        continuous = False
    else:
        logger.error('invalid model type {}'.format(model))
        sys.exit(1)

    if scale_data:
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test)

    logger.debug('Innermost Loop: CV of hyperparameters for this fold')
    clf = cv_loop(model_clf, hyperparams, X_train, y_train) # returns 1 best clf

    # collect all the best hyperparameters found in the cv loop
    for hp in hyperparams:
        hp_dict[hp].append(clf.best_estimator_.get_params()[hp])

    # find out most frequent hyperparameters during cross-val
    for hp in hyperparams:
        hp_mode[hp] = mode(hp_dict[hp])[0][0]
    logger.debug('most frequent hp: {}'.format(hp_mode))

    # collect test / training data stats
    if continuous:
        # ignore p values returned by pearsonr
        r_train = stats.pearsonr(clf.predict(X_train), y_train)[0]
        r_test = stats.pearsonr(clf.predict(X_test), y_test)[0]
    else:
        r_train = np.mean(clf.predict(X_train) == y_train)
        r_test = np.mean(clf.predict(X_test) == y_test)

    R2_train = clf.score(X_train, y_train)
    R2_test = clf.score(X_test, y_test)
    MSE_train = mse(clf.predict(X_train), y_train)
    MSE_test = mse(clf.predict(X_test), y_test)

    logger.debug('predictions,true values\n{}'.format(np.vstack((clf.predict(X_test),  y_test))))

    # check feature importance (QC for HC importance)
    # for fid in np.arange(10):
    #     model_clf.fit(X_train[fid],y_train[fid])
    #     feat_imp = model_clf.feature_importances_
    #     print('\nfid: {} r: {}'.format(fid, zip(*CV_r_valid)[0][fid]))
    #     print(feat_imp[70:], np.argsort(feat_imp)[70:])
    return {'r_train':   r_train,
            'r_test':    r_test,
            'R2_train':  R2_train,
            'R2_test':   R2_test,
            'MSE_train': MSE_train,
            'MSE_test':  MSE_test,
            'hp_dict':   hp_dict,
            'pred_scores': clf.predict(X_test),
            'real_scores': y_test}


def cluster(X, y, plot=None, n_clust=2):
    """
    Creates a distance matrix out of the input matrix Y. Clustering is run on
    this matrix using hierarchical clustering (Ward's algorithm). The data is
    ploted, and the variables in X are shown for all groups in each cluster.
    """
    # hierarchical clustering
    fig = plt.figure()
    axd = fig.add_axes([0.09,0.1,0.2,0.8])
    axd.set_xticks([])
    axd.set_yticks([])
    link = sch.linkage(X, method='ward')
    clst = sch.fcluster(link, n_clust, criterion='maxclust')

    if plot:
        dend = sch.dendrogram(link, orientation='right')
        idx = dend['leaves']
        X = utils.reorder(X, idx, symm=False)
        axm = fig.add_axes([0.3,0.1,0.6,0.8])
        im = axm.matshow(X, aspect='auto', origin='lower', cmap=plt.cm.Reds, vmin=0.3, vmax=0.6)
        axm.set_xticks([])
        axm.set_yticks([])
        axc = fig.add_axes([0.91,0.1,0.02,0.8])
        plt.colorbar(im, cax=axc)
        plt.savefig(os.path.join(plot, 'xbrain_clusters.pdf'))
        plt.close()

        # create seaborn dataframe
        y = standardize(y)
        df = np.hstack((np.atleast_2d(y).T, np.atleast_2d(clst).T))
        df = pd.DataFrame(data=df, columns=['y', 'cluster'])
        df = pd.melt(df, id_vars=['cluster'], value_vars=['y'])

        # plot clinical variables by cluster
        b = sns.boxplot(x="variable", y="value", hue="cluster", data=df, palette="Set3")
        for item in b.get_xticklabels():
            item.set_rotation(45)
        sns.plt.savefig(os.path.join(plot, 'xbrain_cluster_box.pdf'))
        sns.plt.close()
        distributions(y, clst, os.path.join(plot, 'xbrain_cluster_distributions.pdf'))

        # get mean and SD of clinical variables
        means = np.zeros((X.shape[1], len(np.unique(clst))))
        stds  = np.zeros((X.shape[1], len(np.unique(clst))))
        for x in range(X.shape[1]):
            for c in np.unique(clst):
                means[x, c-1] = np.mean(X[clst == c, x])
                stds[x, c-1] = np.std(X[clst == c, x])
        means = ((means.T - means.mean(axis=1))).T

    return clst, idx, means


def distributions(y, clst, plot):
    """Plots data distribution by cluster."""
    unique = np.unique(clst)
    n = len(unique)
    for i in unique:
        plt.subplot(1, n, i)
        # Plot a kernel density estimate and rug plot
        sns.distplot(y[clst == i], hist=False, rug=True, color="r")
    sns.plt.savefig(plot)
    sns.plt.close()


def cluster2(X, plot):
    """
    Hierarchical clustering of the rows in X (subjects). Uses Ward's algorithm.
    """
    fig = plt.figure()
    axd = fig.add_axes([0.09,0.1,0.2,0.8])
    axd.set_xticks([])
    axd.set_yticks([])

    X = np.corrcoef(X)

    link = sch.linkage(X, method='ward')
    clst = sch.fcluster(link, 2, criterion='maxclust')
    dend = sch.dendrogram(link, orientation='right')
    idx = dend['leaves']
    X = X[idx, :]
    X = X[:, idx]

    axm = fig.add_axes([0.3,0.1,0.6,0.8])
    im = axm.matshow(X, aspect='auto', origin='lower', cmap=plt.cm.Reds, vmin=0, vmax=0.5)
    axm.set_xticks([])
    axm.set_yticks([])
    axc = fig.add_axes([0.91,0.1,0.02,0.8])
    plt.colorbar(im, cax=axc)
    plt.show()
    plt.savefig(os.path.join(plot, 'corr.pdf'))

    return clst


