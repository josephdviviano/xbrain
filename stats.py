#!/usr/bin/env python
"""Routines for relating neural activity with clinical variables."""

from copy import copy
import collections
import logging
import os, sys, glob
import random
import re
import string

from scipy import linalg
from scipy.optimize import curve_fit, linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from scipy.stats import lognorm, randint, uniform, mode, spearmanr
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, calinski_harabaz_score, silhouette_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import scale, LabelEncoder, label_binarize, LabelBinarizer, StandardScaler
from sklearn.svm import LinearSVC

# gets rid of ugly warnings related to MPL / SEABORN...
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

# force matplotlib to ignore xwindows
import matplotlib
matplotlib.use('Agg')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns

import correlate as corr
import rcca as rcca
import utils as utils

logger = logging.getLogger(__name__)

def powers(n, exps):
    """returns a list of n raised to the powers in exps"""
    ns = []
    for exp in exps:
        ns.append(n**exp)
    return(np.array(ns))


def classify(X_train, X_test, y_train, y_test, method, output):
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

    n_features = X_train.shape[1]
    hp_dict = collections.defaultdict(list)

    # use AUC score for unbalanced methods b/c outliers are more interesting
    if method == 'anomaly':
        scoring = 'roc_auc'
        model = 'RIF'
    elif method == 'ysplit':
        scoring = 'f1_macro'
        model = 'SVC'
    else:
        scoring = 'f1_macro'
        model = 'SVC'

    plot_X(X_train, os.path.join(output, 'xbrain_X_test-vs-train.pdf'), X2=X_test)

    # settings for the classifier
    if model == 'Logistic':
        clf_mdl = LogisticRegression(class_weight='balanced', verbose=1)
        clf_hp = {'C': config['classify']['logistic']['C'],
                  'penalty': ['l1', 'l2']}
        scale_data = True
    elif model == 'SVC':
        clf_mdl = LinearSVC(max_iter=config['classify']['SVC']['max_iter'],
                            tol=config['classify']['SVC']['tol'],
                            class_weight='balanced')
        clf_hp = {'C': powers(2, range(config['classify']['SVC']['C'][0],
                                       config['classify']['SVC']['C'][1]))}
        scale_data = True
    elif model == 'RFC':
        clf_mdl = RandomForestClassifier(n_jobs=6, class_weight='balanced',
            n_estimators=config['classify']['RF']['n_estimators'])
        clf_hp = {'min_samples_split': np.round(np.linspace(n_features*0.025, n_features*0.2, 10)),
                  'max_depth': np.array([None, 2, 4, 6, 8, 10]),
                  'criterion': ['gini', 'entropy']}
        scale_data = False
    elif model == 'RIF':
        pct_outliers = len(np.where(y_train == -1)[0]) / float(len(y_train))
        clf_mdl = IsolationForest(n_jobs=6, n_estimators=1000, contamination=pct_outliers, max_samples=n_features)
        scale_data = False

    if scale_data:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # perform randomized hyperparameter search to find optimal settings
    if method == 'anomaly':
        clf = clf_mdl
        clf.fit(X_train)
        hp_dict = {'none': 'none'}
    else:

        logger.debug('Inner Loop: Classification using {}'.format(model))
        clf = GridSearchCV(clf_mdl, clf_hp, scoring=scoring, verbose=0)
        clf.fit(X_train, y_train)

        # record best classification hyperparameters found
        for hp in clf_hp:
            hp_dict[hp].append(clf.best_params_[hp])
        logger.debug('Inner Loop: Hyperparameters: {}'.format(hp_dict))
        logger.info('Inner Loop: maximum CV score ({}): {}'.format(scoring, np.max(clf.cv_results_['mean_test_score'])))

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    # make coding of anomalys like other classifiers
    if method == 'anomaly':
        y_train_pred[y_train_pred == -1] = 0
        y_test_pred[y_test_pred == -1] = 0

    logger.debug('test: {}\npred: {}'.format(y_test, y_test_pred))

    # collect performance metrics
    acc = (accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred))
    rec = (recall_score(y_train, y_train_pred, average='macro'),
           recall_score(y_test, y_test_pred, average='macro'),
           recall_score(y_train, y_train_pred, average='micro'),
           recall_score(y_test, y_test_pred, average='micro'))
    prec = (precision_score(y_train, y_train_pred, average='macro'),
            precision_score(y_test, y_test_pred, average='macro'),
            precision_score(y_train, y_train_pred, average='micro'),
            precision_score(y_test, y_test_pred, average='micro'))
    f1 = (f1_score(y_train, y_train_pred, average='macro'),
          f1_score(y_test, y_test_pred, average='macro'),
          f1_score(y_train, y_train_pred, average='micro'),
          f1_score(y_test, y_test_pred, average='micro'))

    # transforms labels to label indicator format, needed for multiclass
    lb = LabelBinarizer()
    lb.fit(y_train)
    # AUC only works if there is more than one class in y
    try:
        auc = (roc_auc_score(lb.transform(y_train), lb.transform(y_train_pred), average='macro'),
               roc_auc_score(lb.transform(y_test), lb.transform(y_test_pred), average='macro'),
               roc_auc_score(lb.transform(y_train), lb.transform(y_train_pred), average='micro'),
               roc_auc_score(lb.transform(y_test), lb.transform(y_test_pred), average='micro'))
    except:
        auc = (0, 0, 0, 0)

    logger.info('TRAIN: confusion matrix\n{}'.format(confusion_matrix(y_train, y_train_pred)))
    logger.info('TEST:  confusion matrix\n{}'.format(confusion_matrix(y_test, y_test_pred)))

    return {'accuracy': acc,
            'recall' : rec,
            'precision': prec,
            'f1': f1,
            'auc': auc,
            'y_test_pred': y_test_pred,
            'n_features_retained': n_features,
            'hp_dict': hp_dict}


def feature_select(X_train, y_train, X_test, n=0):

    if not n:
        n = int(round(X_train.shape[0]*0.25))

    # prune features with 0 variance
    n_before = X_train.shape[1]
    sel = VarianceThreshold()
    sel.fit(X_train)
    X_train = sel.transform(X_train)
    X_test = sel.transform(X_test)
    logger.debug('retained {}/{} features with non-zero variance'.format(X_train.shape[1], n_before))

    # use random forest for feature selection using shallow trees. Inspired by
    # "Variable selection using Random Forests" Genuer R et al. 2012.
    n_before = X_train.shape[1]
    logger.info('random forest feature selection: features={}, samples={}'.format(
        X_train.shape[1], X_train.shape[0]))

    # minimum number of samples in a node that can be split
    # larger values = shallower trees
    mtry = int(round(X_train.shape[1] * 0.33))
    n_trees = 5000
    test = 0
    while test <= n:
        logger.info('training random forest with n_trees={}, mtry={}'.format(
            n_trees, mtry))
        dim_mdl = SelectFromModel(
                      RandomForestClassifier(n_jobs=6, class_weight='balanced',
                          n_estimators=n_trees, min_samples_split=mtry))
        dim_mdl.threshold = 0
        dim_mdl.fit(X_train, y_train)

        importances = dim_mdl.estimator_.feature_importances_
        test = len(np.where(importances)[0]) # number of nonzero importances
        logger.info('{} features found with {}, target features={}'.format(
            test, mtry, n))
        mtry = int(round(mtry * 0.5)) # reduce number of features required to split a node

    # set the threshold for n features
    importances.sort()
    threshold = importances[-n]
    dim_mdl.threshold = threshold
    logger.debug('random forest feature selection threshold {}'.format(threshold))
    X_train = dim_mdl.transform(X_train)
    X_test = dim_mdl.transform(X_test)
    logger.info('random forest retained {}/{} features'.format(n, n_before))

    return(X_train, X_test)


def match_labels(a, b):
    """
    Returns mappins from a to b that minimizes the distance between the input
    label vectors. Inputs must be the same length. The unique values from a + b
    are appended to a + b to ensure that both vectors contain all unique values.

    For details see section 2 of Lange T et al. 2004.

    E.g,
    a = [1,1,2,3,3,4,4,4,2]
    b = [2,2,3,1,1,4,4,4,3]
    optimal: 1 -> 2; 2 -> 3; 3 -> 1; 4 -> 4
    returns:
        1 2
        2 3
        3 1
        4 4

    Inspired by http://things-about-r.tumblr.com/post/36087795708/matching-clustering-solutions-using-the-hungarian
    """
    if len(a) != len(b):
        raise ValueError('length of a & b must be equal')

    ids_a = np.unique(a)
    ids_b = np.unique(b)

    # in some cases, a and b do not have the same number of unique entries. This
    # can happen if one of the two are predicted labels, and the data they were
    # predicted from had some very small classes which constitute outliers, and
    # are not predicted in the held out data. D should still contain an entry
    # for this unlikely class, and our mapping should account for it. To
    # facilitate this, we append all unique values from both a and b to each, so
    # a and b are garunteed to have at least one entry from all unique values
    n = max(len(ids_a), len(ids_b)) # may not be equal
    D = np.zeros((n, n)) # distance matrix
    a = np.hstack((np.hstack((a, ids_a)), ids_b)) # ensures no missing values
    b = np.hstack((np.hstack((b, ids_a)), ids_b)) #

    # constructs the distance matrix between a and b with appended values
    for x in np.arange(n):
        for y in np.arange(n):
            idx_a = np.where(a == x)[0]
            idx_b = np.where(b == y)[0]
            n_int = len(np.intersect1d(idx_a, idx_b))
            # distance = (# in cluster) - 2*sum(# in intersection)
            D[x,y] = (len(idx_a) + len(idx_b) - 2*n_int)

    # permute labels w/ minimum weighted bipartite matching (hungarian method)
    idx_D_x, idx_D_y = linear_sum_assignment(D)
    mappings = np.hstack((np.atleast_2d(idx_D_x).T, np.atleast_2d(idx_D_y).T))

    return(mappings)


def cluster_stability(X, k, n=100):
    """
    Estimates the stability of a given clustering solution (k) by iteratively
    clustering half of the dataset without replacement, and training a
    classifier to propogate these clusters to the held-out data. The agreement
    between these propogated labels and the labels found via clustering is
    reported as the stability (averaged over n iterations).

    See Stability-based validation of clustering solutions. Lange T et al. 2004.
    """
    # store the stability measures for each iteration
    instabilities = np.empty(n)
    instabilities[:] = np.nan

    # hierarchical clustering: ward's method, euclidean distance
    clst = AgglomerativeClustering(n_clusters=k)

    for i in np.arange(n):
        # split data into two random samples
        idx = np.random.permutation(np.arange(X.shape[0]))
        idxa = idx[1:len(idx)/2]
        idxb = idx[len(idx)/2:]
        Xa = X[idxa, :]
        Xb = X[idxb, :]

        # cluster each split
        clst.fit(Xa)
        ya = clst.labels_
        clst.fit(Xb)
        yb = clst.labels_

        # use LDA mapping learned on Xa to predict labels of Xb
        yp = project_labels(Xa, ya, Xb)

        # map yp to match yb, then compute distance
        mappings = match_labels(yp, yb)
        yp_out = np.zeros(len(yp))

        for c in np.arange(k):
            idx_map = np.where(yp == c)
            yp_out[idx_map] = mappings[c, 1]

        # low if cluster solutions are similar
        instabilities[i] = pdist(np.vstack((yp_out, yb)), metric='hamming')

    # normalize instability by that expected by chance for this k
    instability = np.nanmean(instabilities) / (1-(1/k))

    return instability


def project_labels(X_train, y_train, X_test):
    """projects labels onto test set from an LDA mapping from training data"""
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    return(lda.predict(X_test))


def reduce_X_by_y(X, y, p=None, n=None):
    """
    Finds features in X that have a significant searman's rank correlation with
    at least one of the variables in y (using uncorrected p values).
    If p is defined, we keep all features less than p. Else, if n is defined, we
    keep the n smallest p values. Returns an index of those feautres in X.
    """
    # reduce X using the spearman rank correlation between X and y using batches
    n_features = X.shape[1]
    n_tests = y.shape[1]
    idx = np.zeros(n_features, dtype=bool)
    p_vals = []

    # calculate spearmans rho in batch_size feature chunks
    batch_size = 5000
    batches = np.arange(0, n_features, batch_size)
    if batches[-1] != n_features:
        batches = np.append(batches, n_features)

    # p values of the rs between each X and all ys
    for i in range(1, len(batches)):
        ps = spearmanr(X[:, batches[i-1]:batches[i]], y)[1]
        p_vals.append(ps[-n_tests:, 0:batches[i]-batches[i-1]])

    # we only care about the smallest p value for each set of tests
    p_vals = np.min(np.hstack(p_vals), axis=0)

    if p:
        sig = np.where(p_vals <= p)[0]
    elif n:
        sig = np.argsort(p_vals)[:n]
    else:
        raise Exception('neither p nor n are defined')

    idx[sig] = np.bool(1)
    logger.info('reduced X to {} variables'.format(sum(idx)))

    return(idx)


def estimate_biotypes(X, y, y_names, output, k=None, shuffle=False):
    """
    Finds features in X that have a significant spearman's rank correlation
    with at least 1 of the variables in y (uncorrected p < 0.005). The reduced
    feature set X_red is then used to estimate the optimal number of cannonical
    variates that represent the mapping between X_red and y using cross
    validation.

    If k is defined, uses this as the optimal clustering number instead of via
    the stability analysis.

    Next, this estimates the optimal number of clusters in the CCA
    representation of the data.

    Returns the indicies of the reduced features set, the number of cannonical
    variates found, the optimal number of clusters, and the cannonical variates.
    """
    # replace all NaNs with 0's for this analysis
    X[np.isnan(X)] = 0

    # only run CCA on the features of X related in some way to y
    # found to greatly improve the performance of CCA
    idx = reduce_X_by_y(X, y, p=0.005)
    X_red = X[:, idx]

    # shuffles the columns and rows of X_train for null experiments
    if shuffle:
        np.random.shuffle(X_red.T)
        np.random.shuffle(X_red)

    # use regularized CCA to determine the optimal number of cannonical variates
    logger.info('biotyping: cannonical correlation 3-fold cross validation to find brain-behaviour mappings')
    regs = config['biotype']['regs']
    numCCs = np.arange(config['biotype']['n_ccs'][0], config['biotype']['n_ccs'][1])
    cca = rcca.CCACrossValidate(numCCs=numCCs, regs=regs, numCV=3, verbose=False)
    cca.train([X_red, y])
    n_best_cc = cca.best_numCC
    comps = cca.comps[0] # components found in X
    plot_biotype_clusters(comps, os.path.join(output, 'xbrain_biotype_clusters.pdf'))

    # estimate number of clusters by maximizing cluster quality criteria
    if not k:
        logger.debug('estimating n biotypes via stability analysis')
        clst_score = np.zeros(18)
        clst_tests = np.array(range(2,20))
        for i, n_clst in enumerate(clst_tests):
            clst_score[i] = cluster_stability(comps, n_clst)
        target = np.min(clst_score) # minimize instability
        n_best_clst = clst_tests[clst_score == target][0]
        plot_n_cluster_estimation(clst_score, clst_tests,
            os.path.join(output, 'xbrain_biotype_n_cluster_estimation.pdf'))
    else:
        logger.debug('using defined n_biotypes: {}'.format(k))
        n_best_clst = k

    clst = AgglomerativeClustering(n_clusters=n_best_clst)
    clst.fit(comps)
    clst_labels = clst.labels_

    # calculate connectivity centroids for each biotype
    clst_centroids = np.zeros((n_best_clst, X.shape[1]))
    for n in range(n_best_clst):
        clst_centroids[n, :] = np.mean(X[clst_labels == n, :], axis=0)

    logger.info('biotyping: found {} cannonical variates, {} n biotypes'.format(n_best_cc, n_best_clst))

    # save biotype information
    np.savez_compressed(os.path.join(output, 'xbrain_biotype.npz'),
        reg = cca.best_reg,
        n_cc = n_best_cc,
        n_clst = n_best_clst,
        clusters = clst_labels,
        cancorrs = cca.cancorrs,
        centroids = clst_centroids,
        comps_X = cca.comps[0],
        comps_y = cca.comps[1],
        ws_X = cca.ws[0],
        ws_y = cca.ws[1],
        idx_X = idx,
        X = X,
        y = y,
        y_names = y_names,
        cca = cca)
    mdl = np.load(os.path.join(output, 'xbrain_biotype.npz'))

    return mdl


def biotype(X_train, X_test, y_train, mdl, shuffle=False):
    """
    X is a ROI x SUBJECT matrix of features (connectivities, cross-brain
    correlations, etc.), and y N by SUBJECT matrix of outcome measures (i.e.,
    cognitive variables, demographics, etc.).

    Will decompose X_train and y_train into n_cc cannonical variates (n is
    determined via cross-validation in estimate_biotypes). The previously
    estimated regularization parameter reg will be used.

    mdl is a dictionary containing biotype information (see estimate_biotypes).

    CCA will only be run on the features defined in idx.

    We want to estimate the number of variates before training the model, and
    then try to find the same number of variates for each fold. Use
    estimate_cca to find n and idx.

    The pipeline is as follows:

    1) Use connonical correlation to find a n_cc dimentional mapping between
       the eeduced feature matrix X_train_red and y_train.
    2) Cluster the subjects using Ward's hierarchical clustering into biotypes,
       generating biotype labels (the new y_train).
    3) Train a linear discriminate classifier on X_train_red and the cluster
       labels, and then run this model on X_test, to produce biotype labels for
       the test set (the new y_test).

    This returns y_train and y_test, the discovered biotypes for classification.
    """
    # get information from biotype model
    reg = mdl['reg']
    n_clst = mdl['n_clst']
    n_cc = mdl['n_cc']
    idx_len = np.sum(mdl['idx_X'])

    # replace all NaNs with 0's for this analysis
    X_train[np.isnan(X_train)] = 0
    X_test[np.isnan(X_test)] = 0
    X_train[np.isinf(X_train)] = 0
    X_test[np.isinf(X_test)] = 0

    # only run CCA on the features of X related in some way to y
    # found to greatly improve the performance of CCA
    idx = reduce_X_by_y(X_train, y_train, n=idx_len)
    X_train_red = X_train[:, idx]
    X_test_red = X_test[:, idx]

    # shuffles the columns and rows of X_train for null experiments
    if shuffle:
        np.random.shuffle(X_train_red.T)
        np.random.shuffle(X_train_red)

    # use regularized CCA to find brain-behaviour mapping from the training set
    logger.info('biotyping training set')
    cca = rcca.CCA(numCC=n_cc, reg=reg, verbose=False)
    cca.train([X_train_red, y_train])
    comps = cca.comps[0] # [0] uses comps from X, [1] uses comps from y

    # cluster these components to produce n_clst biotypes, y_train
    clst = AgglomerativeClustering(n_clusters=n_clst)
    clst.fit(comps)
    y_train = clst.labels_

    # use LDA to predict the labels of the test set, y_test
    y_test = project_labels(X_train_red, y_train, X_test_red)

    return y_train, y_test


def get_states(d_rs, k=5):
    """
    Accepts a ROI x TIMEPOINT dynamic connectivity matrix, and returns K states
    as determined by K-means clustering (ROI x STATE).

    Uses a slighly less accurate implementation of kmeans designed for very
    large numbers of samples: D. Sculley, Web-Scale K-Means Clustering.
    """
    clf = MiniBatchKMeans(n_clusters=k)
    logger.debug('running kmeans on X {}, k={}'.format(d_rs.shape, k))
    clf.fit(d_rs.T)
    return(clf.cluster_centers_.T)


def fit_states(d_rs, states):
    """
    Accepts a ROI x TIMEPOINT dynamic connectivity matrix, and a ROI x STATE
    matrix (composed of the outputs from get_states), and computes the
    regression coefficients for each time window against all input states.
    Returns the sum of the coefficients across all time windows for each state.
    Could be thought of as a measure of how much relative time this subject
    spent in each state during the scan.
    """
    d_rs = utils.clean(d_rs)
    clf = LinearRegression()
    clf.fit(d_rs, states)
    return(np.mean(clf.coef_, axis=1))


def r_to_z(R):
    """Fischer's r-to-z transform on a matrix (elementwise)."""
    return(0.5 * np.log((1+R)/(1-R)))


def r_to_d(R):
    """Converts a correlation matrix R to a distance matrix D."""
    return(np.sqrt(2*(1-R)))


def standardize(X):
    """z-scores each column of X."""
    return((X - X.mean(axis=0)) / X.std(axis=0))


def standardize_by_group(X, labels, group):
    """z-scores each column of X by the mean and standard deviation of group."""
    assert(X.shape[0] == len(labels))
    idx = np.where(labels == group)[0]
    X_group_mean = X[idx, :].mean(axis=0)
    X_group_std = X[idx, :].std(axis=0)
    return((X - X_group_mean) / X_group_std)


def gauss(x, *p):
    """Model gaussian to fit to data."""
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


def find_outliers(y, output, n_sigma=2):
    """
    Assumes y is the mixture of scores drawn from a normal distribution and
    some unusual, unknown distribution. Fits a gaussian curve to y, and finds
    that curve's mean and standard deviation. This model assumes that only 2.5%
    of the data should fall below the -2*sigma line. Finds the actual percentage
    of datapoints below that line, subtracts 2.5% from it (to account for the
    expected percentage), and then flags all of those data points as outliers
    with negative -1. Normal values are set to 1. Returns this modified y
    vector, and the percentage of the data that are considered outliers.
    """
    binned_curve, bin_edges = np.histogram(y, bins=len(y)/10, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
    # initial curve search values
    p0 = [1., 0., 1.]
    coeff, var_matrix = curve_fit(gauss, bin_centres, binned_curve, p0=p0)

    mean = coeff[1]
    sd = coeff[2]

    # interested in more than expected number of values below -2 SD
    null_cutoff = mean - n_sigma*sd
    #null_outliers_pct = 0.025 # 2.5% of data expected below 2 sd
    real_outliers_pct = len(np.where(y < null_cutoff)[0]) / float(len(y))
    #diff_outliers_pct = real_outliers_pct - null_outliers_pct
    if not utils.is_probability(real_outliers_pct):
        logger.error('auto-found probability is invalid: {}, should be [0,1]'.format(real_outliers_pct))
        logger.error('this likely means your y distribution is gaussian. try running with a set cutoff in diagnostics mode to confirm.')
        sys.exit(1)
    diff_cutoff = np.percentile(y, real_outliers_pct*100)

    y_outliers = copy(y)
    y_outliers[y <= diff_cutoff] = 0
    y_outliers[y > diff_cutoff] = 1

    logger.info('auto-partitioning y at the {}th percentile (non-gaussian outliers)'.format(real_outliers_pct*100))
    plot_gauss_fit(bin_centres, binned_curve, diff_cutoff, coeff,
        os.path.join(output, 'xbrain_y_outlier_fit.pdf'))

    return y_outliers


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

    n = D.shape[0]
    o = np.ones((n, 1))
    I = np.identity(n) - (1/float(n))*o.dot(o.T)
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


def mdmr(X, Y, method='corr'):
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

    if method == 'corr':
        R = np.corrcoef(Y)   # correlation distance between each cross-brain correlation vector
        D = r_to_d(R)        # distance matrix of correlation matrix
    elif method == 'euclidean':
        D = squareform(pdist(Y, 'euclidean'))

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


def pca_reduce(X, n=1, pct=0, X2=False):
    """
    Uses PCA to reduce the number of features in the input matrix X. n is
    the target number of features in X to retain after reduction. pct is the
    target amount of variance (%) in the original matrix X to retain in the
    reduced feature matrix. When n and pct disagree, compresses the feature
    matrix to the larger number of features. If X2 is defined, the same
    transform learned from X is applied to X2.
    """
    if not utils.is_probability(pct):
        raise Exception('pct should be a probability (0-1), is {}'.format(pct))

    clf = PCA()
    clf = clf.fit(X)
    cumulative_var = np.cumsum(clf.explained_variance_ratio_)
    n_comp_pct = np.where(cumulative_var >= pct)[0][0] + 1 # fixes zero indexing

    if n > n_comp_pct:
        pct_retained = cumulative_var[n-1]
        cutoff = n
    else:
        pct_retained = cumulative_var[n_comp_pct-1]
        cutoff = n_comp_pct

    logger.info('X {} reduced to {} components, retaining {} % of variance'.format(X.shape, cutoff, pct_retained))

    # reduce X to the defined number of components
    clf = PCA(n_components=cutoff)
    clf.fit(X)
    X_transformed = clf.transform(X)

    # sign 1st PC of X if inverse to mean of X
    if cutoff == 1:
        X_transformed = sign_flip(X_transformed, X)

    # if X2 is defined, apply the transform learnt from X to X2 as well
    if np.any(X2):
        X2_transformed = clf.transform(X2)

        if cutoff == 1:
            X2_transformed = sign_flip(X2_transformed, X2)

        logger.debug('PCA transform learned on X applied to X2')
        return(X_transformed, X2_transformed)

    return(X_transformed)


def sign_flip(X_transformed, X):
    """
    X_transformed a 1D vector representing the top PC from X. This applies a
    sign flip to X_transformed if X_transformed is anti-correlated with the mean
    of X. This is important particularly for compressing the y variables, where
    we want to retain high (good) and low (scores), and flipping these would
    change our intrepretation of the statistics.
    """
    X_transformed = X_transformed.flatten()
    corr = np.corrcoef(np.vstack((X_transformed, np.mean(X, axis=1))))[0,1]
    if corr < 0:
        X_transformed = X_transformed * -1

    return(X_transformed)


def make_classes(y, target_group):
    """transforms label values for classification, including target_group"""
    le = LabelEncoder()
    le.fit(y)
    logger.debug('y labels {} transformed to {}'.format(
        le.classes_, np.arange(len(le.classes_))))
    y = le.transform(y)

    # TODO fix ugly type gymnatsitcs
    if target_group >= 0:
        try:
            target_group = int(np.where(target_group == le.classes_)[0][0])
        except:
            target_group = int(np.where(int(target_group) == le.classes_)[0][0])

    return(y, target_group)


def covary(X, X_cov):
    """
    For each column in X, fits all covariates in X_cov, and keeps the residuals.
    """
    clf = LinearRegression(normalize=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_cov = scaler.fit_transform(X_cov)

    # predict all brain connections given the covariates of no interest
    # calc residuals as y - yhat
    if len(X_cov.shape) == 1:
        clf.fit(X_cov.reshape(-1, 1), X)
        X -= clf.predict(X_cov.reshape(-1, 1))
    else:
        clf.fit(X_cov, X)
        X -= clf.predict(X_cov)

    return(X)


def diagnostics(X_train, X_test, y_train, y_test, y_train_raw, output):
    """
    A pipeline for assessing the inputs to the classifier. Runs on
    a single fold.

    + Loads X and y. If y_raw has multiple preditors, the top PC is calculated.
    + Plots a distribution of y_raw, compressed to 1 PC.
    + Saves a .csv with this compressed version of y_raw.
    + Plots the top 3 PCs of X_train, with points colored by group y_train. This
      plot should have no trivial structure.
    + Plots a hierarchical clustering of X_train.
    + Uses MDMR to detect relationship between y_raw and X_train.
    + Runs iterative classification of the training data using logistic
      regression and l1 regularization with varying levels of C.
    """
    logger.info('running diagnostics on neural and cognitive data')
    logger.debug('compressing y to 1d')
    if len(y_train_raw.shape) == 2 and y_train_raw.shape[0] > 1:
        y_1d = copy(pca_reduce(y_train_raw))
    else:
        y_1d = copy(y_train_raw)

    logger.debug('plotting y (1d) before generating classes')
    plot_distributions(y_1d.T, os.path.join(output, 'xbrain_y_dist.pdf'))

    logger.debug('plotting X_train and X_test distributions')
    plot_distributions(X_train, os.path.join(output, 'xbrain_X_dist.pdf'), X2=X_test)

    logger.debug('printing top 3 PCs of X, colour coding by y group')
    plot_pcs(X_train, y_train, os.path.join(output, 'xbrain_X_PCs.pdf'))

    logger.debug('saving copies of X and y')
    np.save(os.path.join(output, 'xbrain_y.npy'), y_1d)
    np.save(os.path.join(output, 'xbrain_X.npy'), X_train)

    logger.debug('plotting hierarchical clustering of the feature matrix X')
    plot_clusters(X_train, os.path.join(output, 'xbrain_X_clusters.pdf'))

    # TODO: significance test is high even if explained variance is very low.
    #       double check everything, and evaluate utilitiy of such a test for
    #       classification (when the noise level is so high, SVM etc gets
    #       bamboozled).
    # use MDMR to find a relationship between the X matrix and all y predictors
    # broken -- need to go over matrix transpositions...
    #F, F_null, v = mdmr(y_train_raw, X_train, method='euclidean')
    #thresholds = sig_cutoffs(F_null, two_sided=False)
    #if F > thresholds[1]:
    #    logger.info('mdmr: relationship detected: F={} > {}, variance explained={}'.format(F, thresholds[1], v))
    #else:
    #    logger.warn('mdmr: no relationship detected, variance explained={}'.format(v))

    logger.debug('overfitting tests over range of sparsities, logistic regression')
    Cs = powers(2, np.arange(-7, 7, 0.5))
    features, train_score, test_score = [], [], []
    for i, C in enumerate(Cs):
        logger.debug('testing C={}, test={}/{}'.format(C, i+1, len(Cs)))
        # fit a logistic regression model with l1 penalty (for sparsity)
        mdl = LogisticRegression(n_jobs=6, class_weight='balanced', penalty='l1', C=C)
        mdl.fit(X_train, y_train)
        # use non-zero weighted features to reduce X
        dim_mdl = SelectFromModel(mdl, prefit=True)
        X_train_pred = mdl.predict(X_train)
        X_test_pred= mdl.predict(X_test)
        features.append(dim_mdl.transform(X_train).shape[1])
        train_score.append(f1_score(y_train, X_train_pred, average='macro'))
        test_score.append(f1_score(y_test, X_test_pred, average='macro'))

    # plot test and train scores against number of features retained
    plt.plot(train_score, color='black')
    plt.plot(test_score, color='red')
    plt.xticks(np.arange(len(features)), features)
    plt.legend(['train', 'test'])
    plt.xlabel('n features retained')
    plt.ylabel('F1 score (macro)')
    plt.savefig(os.path.join(output, 'xbrain_overfit-analysis.pdf'))


def plot_pcs(X, y, output):
    """
    Takes the top 3 PCs from the data matrix X, and plots them. y is used to
    color code the data. No obvious grouping or clustering should be found. The
    presence of such grouping suggests a strong site, sex, or similar effect.
    """
    clf = PCA(n_components=3)
    X = clf.fit_transform(X)
    fig = plt.figure(1, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.RdBu_r)
    fig.savefig(output)
    plt.close()


def plot_X(X, output, X2=None):
    """
    Plots the cross brain correlation features calculated. Can be used to
    compare features (e.g., hi vs low template, or train vs test matricies) if
    X2 is defined. Negative correlations are nonsense for cross brain
    correlations, and are set to 0 for visualization.
    """
    if X2 is not None:
        X = np.vstack((np.vstack((X, np.ones(X.shape[1]))), X2))

    plt.imshow(X, vmin=-0.5, vmax=0.5, cmap=plt.cm.RdBu_r, interpolation='nearest')
    plt.colorbar()

    if X2 is not None:
        plt.title('X vs X2 feature matricies')
    else:
        plt.title('X feature matrix')

    plt.savefig(output)
    plt.close()


def plot_distributions(X, output, X2=None):
    """
    Plots data distribution of input matrix X. If X2 is defined, constrast the
    distributions of the two matrices.
    """
    # rug plots take forever if there are too many data points
    if len(np.ravel(X)) < 1000:
        rug = True
    else:
        rug = False
    sns.distplot(np.ravel(X), hist=False, rug=rug, color="r")
    if X2 is not None:
        sns.distplot(np.ravel(X2), hist=False, rug=rug, color="black")
    sns.plt.savefig(output)
    sns.plt.close()


def plot_clusters(X, output):
    """
    Plots a simple hierarchical clustering of the feature matrix X. Clustering
    is done using Ward's algorithm. Variables in X are arranged by cluster.
    """
    fig = plt.figure()
    axd = fig.add_axes([0.09,0.1,0.2,0.8])
    axd.set_xticks([])
    axd.set_yticks([])
    link = sch.linkage(X, method='ward')
    dend = sch.dendrogram(link, orientation='right')
    idx = dend['leaves']
    X = utils.reorder(X, idx, symm=False)
    axm = fig.add_axes([0.3,0.1,0.6,0.8])
    im = axm.matshow(X, aspect='auto', origin='lower', cmap=plt.cm.Reds, vmin=-0.5, vmax=0.5)
    axm.set_xticks([])
    axm.set_yticks([])
    axc = fig.add_axes([0.91,0.1,0.02,0.8])
    plt.colorbar(im, cax=axc)
    plt.savefig(output)
    plt.close()


def plot_biotype_X_scatter(mdl, output):
    """scatterplot of X components colored by cluster"""
    fig = plt.figure()
    if mdl['comps_X'].shape[1] >= 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(mdl['comps_X'][:, 0], mdl['comps_X'][:, 1], mdl['comps_X'][:, 2],
            c=mdl['clusters'])
    else:
        ax = fig.add_subplot(111)
        ax.scatter(mdl['comps_X'][:, 0], mdl['comps_X'][:, 1], c=mdl['clusters'])
    fig.savefig(output)
    plt.close()


def plot_biotype_y_loadings(mdl, output):
    """plots relationship between each y component and the submitted y scores"""
    correlations = np.zeros((mdl['y'].shape[1], mdl['comps_y'].shape[1]))
    for i, score in enumerate(mdl['y'].T):
        for j, comp in enumerate(mdl['comps_y'].T):
            correlations[i, j] = np.corrcoef(score, comp)[0,1]
    plt.imshow(correlations, cmap=plt.cm.RdBu_r, vmin=-1, vmax=1)
    plt.yticks(range(len(mdl['y_names'])), mdl['y_names'])
    plt.colorbar()
    plt.savefig(output)
    plt.close()


def plot_biotype_X_conn_loadings(mdl, X, mask, output):
    """
    prints the sum of connectivity loadings per ROI in mask for each canonical
    variate. connectivity values are represented as the average positive
    connectivity with all other ROIs.
    """

    nii = nib.load(mask)
    nii_data = nii.get_data()
    rois = np.unique(nii_data[nii_data > 0])

    # calculate the correlation of each connectivity feature with each component
    correlations = np.zeros((X.shape[1], mdl['comps_X'].shape[1]))
    for i, conn in enumerate(X.T):
        for j, comp in enumerate(mdl['comps_X'].T):
            correlations[i, j] = np.corrcoef(conn, comp)[0,1]

    output_list = []
    for i in range(correlations.shape[1]):
        # copy of input atlas to store correlations (+ve and -ve seperate)
        atlas_corrs_pos = np.zeros(nii_data.shape)
        atlas_corrs_neg = np.zeros(nii_data.shape)

        # reconstruct correlation matrix (now representing relationships between
        # connectivity features in X and the components found for X).
        roi_conns = np.zeros((len(rois), len(rois)))
        idx_triu = np.triu_indices(len(rois), k=1)
        roi_conns[idx_triu] = correlations[:, i]
        roi_conns = roi_conns + roi_conns.T

        # seperate +ve and -ve correlations so we can take mean of each
        roi_conns_pos = np.zeros(roi_conns.shape)
        roi_conns_neg = np.zeros(roi_conns.shape)
        roi_conns_pos[roi_conns >= 0] = roi_conns[roi_conns >= 0]
        roi_conns_neg[roi_conns < 0] = roi_conns[roi_conns < 0]
        roi_conns_pos = np.nanmean(roi_conns_pos, axis=1)
        roi_conns_neg = np.nanmean(roi_conns_neg, axis=1)

        # load these connectivity values into the ROI mask
        for j, roi in enumerate(rois):
            atlas_corrs_pos[nii_data == roi] = roi_conns_pos[j]
            atlas_corrs_neg[nii_data == roi] = roi_conns_neg[j]

        output_list.append(atlas_corrs_pos)
        output_list.append(atlas_corrs_neg)

    # save ROI connectivity correlations per component to nifti
    output_nii = np.stack(output_list, axis=3)
    output_nii = nib.nifti1.Nifti1Image(output_nii, nii.affine, header=nii.header)
    output_nii.update_header()
    output_nii.header_class(extensions=())
    output_nii.to_filename(output)


def plot_biotype_X_stat_loadings(mdl, X, mask, output):
    """prints the stat for each ROI in mask for each canonical variate"""
    nii = nib.load(mask)
    nii_data = nii.get_data()
    rois = np.unique(nii_data[nii_data > 0])

    # calculate the correlation of each stat with each component
    correlations = np.zeros((X.shape[1], mdl['comps_X'].shape[1]))
    for i, stat in enumerate(X.T):
        for j, comp in enumerate(mdl['comps_X'].T):
            correlations[i, j] = np.corrcoef(stat, comp)[0,1]

    output_list = []
    for i in range(correlations.shape[1]):
        # copy of input atlas to store statistics
        #atlas_corrs = np.zeros(nii_data.shape)
        atlas_corrs_pos = np.zeros(nii_data.shape)
        atlas_corrs_neg = np.zeros(nii_data.shape)

        # load stat values into ROI mask
        roi_conns = np.zeros((len(rois)))
        roi_conns_pos = np.zeros(roi_conns.shape)
        roi_conns_neg = np.zeros(roi_conns.shape)

        roi_conns = correlations[:, i]
        roi_conns_pos[roi_conns >= 0] = roi_conns[roi_conns >= 0]
        roi_conns_neg[roi_conns < 0] = roi_conns[roi_conns < 0]

        for j, roi in enumerate(rois):
            #atlas_corrs[nii_data == roi] = roi_conns[j]
            atlas_corrs_pos[nii_data == roi] = roi_conns_pos[j]
            atlas_corrs_neg[nii_data == roi] = roi_conns_neg[j]

        output_list.append(atlas_corrs_pos)
        output_list.append(atlas_corrs_neg)
        #output_list.append(atlas_corrs)

    # save ROI connectivity correlations per component to nifti / cifti
    output_nii = np.stack(output_list, axis=3)
    if nii.header_class == nib.nifti2.Nifti2Header:
        output_nii = nib.nifti2.Nifti2Image(output_nii, nii.affine, header=nii.header)
    else:
        output_nii = nib.nifti1.Nifti1Image(output_nii, nii.affine, header=nii.header)
        output_nii.update_header()
        output_nii.header_class(extensions=())

    output_nii.to_filename(output)


def plot_biotype_cluster_scores(mdl, output):
    """boxplot of each y score from each biotype"""
    # build dataframe for plotting
    db = pd.DataFrame()
    db['id'] = range(mdl['y'].shape[0])        # ID
    db['biotype'] = mdl['clusters']            # grouping variable
    for i in range(mdl['y'].shape[1]):
        db[mdl['y_names'][i]] = standardize(mdl['y'][:, i]) # cognitive score, z scored
    db = pd.melt(db, id_vars=['id', 'biotype'], value_vars=mdl['y_names'].tolist())

    sns.boxplot(x="variable", y="value", hue="biotype", data=db, palette="RdBu")
    sns.plt.savefig(output)
    sns.plt.close()


def plot_biotype_clusters(comps, output):
    """uses hierarchical clustering (ward's method) to show biotypes"""
    sns.clustermap(comps, method='ward', metric='euclidean', col_cluster=False)
    sns.plt.savefig(output)
    sns.plt.close()


def plot_n_cluster_estimation(clst_score, clst_tests, output):
    """Plots the cluster goodness score against the number of clusters"""
    plt.plot(clst_score)
    plt.title('cluster instability analysis')
    plt.ylabel('instability')
    plt.xlabel('number of clusters (k)')
    plt.xticks(range(len(clst_tests)), clst_tests)
    plt.savefig(output)
    plt.close()


def plot_gauss_fit(bin_centres, binned_curve, diff_cutoff, coeff, output):
    """plots a fit of a gaussian curve to the empirical data"""
    fitted_curve = gauss(bin_centres, *coeff)
    plt.plot(bin_centres, binned_curve, color='k', label='y')
    plt.plot(bin_centres, fitted_curve, color='r', label='gaussian fit')
    plt.axvline(x=diff_cutoff, color='k', linestyle='--')
    plt.savefig(output)
    plt.close()


