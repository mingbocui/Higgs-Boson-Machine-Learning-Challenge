import numpy as np
from implementations import *
from evaluation import predict_regression_labels
from proj1_helpers import build_poly
from preprocess import standardize

def test_accuracy(y, tx, w):
    """calculates the final score of a given method"""
    labels = predict_regression_labels(w, tx)
    
    return (labels==y).sum()/len(y)

def compute_log(tx, index_log, mean=[], std=[]):
    """compute the log value of the given features,
    and then standardize the features and return mean and std
    """
    tx_new = np.log10(3+abs(tx[:,index_log]))
    return standardize(tx_new,mean,std)

def compute_theta(tx, index_theta, mean=[],std=[]):
    """compute the cosine value of the angle-like features,
    and then standardize the features and return mean and std
    """
    tx_new = np.cos(tx[:,index_theta])
    return standardize(tx_new,mean,std)

def compute_physics(tx, index_A, index_B, index_C,
                    mean=[], std=[]):
    """add the features that have physics means,
    using the particle mass (index_A = 0) as the weight,
    and then standardize the features and return mean and std
    """
    tx_new = tx[:,index_A] * tx[:,index_B] / tx[:,index_C]
    return standardize(tx_new,mean,std)

def select_best_degrees(y, tx, reg_function=least_squares, max_degree=5):
    """Compute the optimal polynomial expansion degree for each feature individually"""
    degrees = np.ones(tx.shape[1])
    scores = np.zeros(tx.shape[1])

    for i in range(tx.shape[1]):
        for d in range(1, max_degree+1):
            feature_with_degree = build_poly(tx[:,i], d)
            tx_rest = np.delete(tx, i, axis=1)
            feature_with_degree = np.c_[feature_with_degree,tx_rest]

            w, _ = reg_function(y, feature_with_degree)
            score = test_accuracy(y, feature_with_degree, w)
            if score > scores[i]:
                scores[i] = score
                degrees[i] = d
    return degrees

def build_poly_by_feature(tx, degrees):
    """Builds a polynomial expansion based on the given degree for each feature"""
    poly_tempt = np.ones([tx.shape[0],1])
    for idx, degree in enumerate(degrees):
        feature_poly = build_poly(tx[:,idx], int(degree))
        poly_tempt = np.c_[poly_tempt, feature_poly[:,1:]]
    return poly_tempt