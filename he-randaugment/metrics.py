''' Useful metrics '''

import pandas as pd
from os.path import basename, dirname, exists, join, splitext
from matplotlib import pyplot as plt
import os
import numpy as np
import random
from sklearn.metrics import roc_curve, auc, accuracy_score
import pickle
import shutil
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from scipy import stats
from sklearn.metrics import log_loss, roc_auc_score
from scipy import interpolate


# ----------------------------------------------------------------------------------------------------

def map_probabilities(y_pred, label_map):

    if y_pred.shape[1] < 2:
        print('Warning, map_probabilities() should received one-hot encoded vectors. y_pred shape is {s}'.format(s=y_pred.shape), flush=True)

    label_map = {int(k): int(v) for k, v in label_map.items()}
    inv_map = {}
    for k, v in label_map.items():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)

    n_samples = y_pred.shape[0]
    n_source_classes = len(np.unique(list(label_map.keys())))
    n_target_classes = len(np.unique(list(label_map.values())))

    y_pred_target = np.zeros((n_samples, n_target_classes))

    for target_class in range(n_target_classes):

        source_idx = inv_map[target_class]
        y_pred_target[:, target_class] = y_pred[:, source_idx].sum(axis=1)

    return y_pred_target

# ----------------------------------------------------------------------------------------------------

def compute_label_weights(y_true, one_hot=True):

    if one_hot:
        y_true_single = np.argmax(y_true, axis=-1)
    else:
        y_true_single = y_true

    w = np.ones(y_true_single.shape[0])
    for idx, i in enumerate(np.bincount(y_true_single)):
        w[y_true_single == idx] *= 1/(i / float(y_true_single.shape[0]))

    return w

# ----------------------------------------------------------------------------------------------------

def auc_fn(y_true, y_pred):

    try:
        auc_score = roc_auc_score(y_true, y_pred)
    except Exception as e:
        print('AUC could not be computed, returning 0. Exception: %s' % str(e), flush=True)
        auc_score = 0

    return auc_score

# ----------------------------------------------------------------------------------------------------

def accuracy_fn(y_true, y_pred, label_map=None, threshold=0.5):

    # Label map
    if label_map is not None:
        # y_true = map_probabilities(y_true, label_map)
        y_pred = map_probabilities(y_pred, label_map)

    # Thresholding
    y_pred_th = np.array(y_pred > threshold).astype('int')

    # # Label map
    # if label_map is not None:
    #     y_pred_th = np.array([label_map[str(pred)] for pred in y_pred_th])

    acc = accuracy_score(y_true, y_pred_th)

    return acc


# ----------------------------------------------------------------------------------------------------

def accuracy_weighted_fn(y_true, y_pred, label_map=None, threshold=0.5):

    # Label map
    if label_map is not None:
        # y_true = map_probabilities(y_true, label_map)
        y_pred = map_probabilities(y_pred, label_map)

    # Weights
    w = compute_label_weights(y_true, one_hot=True)

    # Thresholding
    y_pred_th = np.array(y_pred > threshold).astype('int')

    # No one-hot
    y_true = np.argmax(y_true, axis=-1)
    y_pred_th = np.argmax(y_pred_th, axis=-1)

    # # Label map
    # if label_map is not None:
    #     y_pred_th = np.array([label_map[str(pred)] for pred in y_pred_th])

    # Score
    acc = accuracy_score(y_true, y_pred_th, sample_weight=w)

    return acc

# ----------------------------------------------------------------------------------------------------

def log_loss_fn(y_true, y_pred):

    return log_loss(y_true, y_pred)

# ----------------------------------------------------------------------------------------------------

def log_loss_weighted_fn(y_true, y_pred):

    w = compute_label_weights(y_true, one_hot=True)

    return log_loss(y_true, y_pred, sample_weight=w)



