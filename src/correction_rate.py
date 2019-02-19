# -*- coding: utf-8 -*-
import numpy as np
from implementations import *
from costs import compute_loss
from evaluation import regresssion_score, logistic_score

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def run_method(y, tx, method, params):
    """Run the wanted method and returns its value"""
    if method == "LS":
        return least_squares(y, tx)
    elif method == "GD":
        return least_squares_GD(y, tx, params["initial_w"], params["max_iters"], params["gamma"])
    elif method == "SGD":
        return least_squares_SGD(y, tx, params["initial_w"], params["max_iters"], params["gamma"])
    elif method == "RR":
        return ridge_regression(y, tx, params["lambda_"])
    elif method == "LR":
        return logistic_regression(y, tx, params["initial_w"], params["max_iters"], params["gamma"])
    elif method == "RLR":
        return reg_logistic_regression(y, tx, params["lambda_"], params["initial_w"], params["max_iters"], params["gamma"])

def cross_validation(y, tx, method, params,threshold=0, k=5, seed=0):
    """Gives the mean RMSE for running the given method with the given parameters"""
    k_indices = build_k_indices(y, k, seed)
    s = np.zeros(k)
    l = np.zeros(k)
    
    weights = []
    scores = []
    for i in range(k):
        # get k'th subgroup in test, others in train
        te_indice = k_indices[i]
        tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == i)]
        tr_indice = tr_indice.reshape(-1)
        y_test = y[te_indice]
        y_train = y[tr_indice]
        tx_test = tx[te_indice]
        tx_train = tx[tr_indice]
        
        w, _ = run_method(y_train, tx_train, method, params)
        weights.append(w)
        
        if method in [ "LR", "RLR" ]:
            l[i] = compute_loss(y_test, tx_test, w, loss_method='sigmoid')
            s[i] = logistic_score(y_test, tx_test, w, threshold=threshold)
        else:
            l[i] = compute_loss(y_test, tx_test, w, loss_method='rmse')
            s[i] = regresssion_score(y_test, tx_test, w, threshold=threshold)
        
    return s, weights, np.mean(l), np.mean(s) 

"""Initial w doesn't matter and max_iters is based on what our computer can handle, so they aren't variable"""
def test_LS(y, tx):
    return cross_validation(y, tx, "LS", {})

def test_GD(y, tx, gamma):
    return cross_validation(y, tx, "GD", {
        "initial_w": np.zeros(tx.shape[1]),
        "max_iters": 100,
        "gamma"    : gamma
        })

def test_SGD(y, tx, gamma):
    return cross_validation(y, tx, "SGD", {
        "initial_w": np.zeros(tx.shape[1]),
        "max_iters": 100,
        "gamma"    : gamma
        })

def test_RR(y, tx, lambda_):
    return cross_validation(y, tx, "RR", {
        "lambda_"  : lambda_
        })

def test_LR(y, tx, gamma):
    return cross_validation(y, tx, "LR", {
        "initial_w": np.zeros(tx.shape[1]),
        "max_iters": 100,
        "gamma"    : gamma
        })

def test_RLR(y, tx, lambda_, gamma):
    return cross_validation(y, tx, "RLR", {
        "initial_w": np.zeros(tx.shape[1]),
        "max_iters": int(100),
        "gamma"    : float(gamma),
        "lambda_"  : float(lambda_),
        })

# 
def print_score(loss_avg, score_avg, scores, method='Ridge regression'):
    dash = '=' * 60
    print(dash)
    print('{:^60s}'.format(method))
    print(dash)
    print('{:<20s} {:.>40.3e}'.format('average loss',loss_avg))
    print('{:<20s} {:.>40.3e}'.format('average accuracy', score_avg))
    print('{:<20s} {:.>40.3e}'.format('max accuracy', np.max(scores)))
    print(dash)
    return