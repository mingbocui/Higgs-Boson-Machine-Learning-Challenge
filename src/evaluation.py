# -*- coding: utf-8 -*-
"""some evaluation functions."""
import numpy as np
from costs import sigmoid

# def predict_labels(weights, data, threshold=-0.001):
#     """Generates class predictions given weights, and a test data matrix"""
#     y_pred = np.dot(data, weights)
#     y_pred[np.where(y_pred <= threshold)] = 0
#     y_pred[np.where(y_pred > threshold)] = 1
#     return y_pred

def predict_regression_labels(weights, data, threshold=0):
    """Generates regression label given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= threshold)] = -1
    y_pred[np.where(y_pred > threshold)] = 1
    return y_pred

def predict_logistic_labels(weights, data, threshold=0.5):
    """Generates logistic classification given weights, and a test data matrix"""
    y_pred = sigmoid(np.dot(data, weights))
    y_pred[np.where(y_pred <= threshold)] = 0
    y_pred[np.where(y_pred > threshold)] = 1
    return y_pred

def regresssion_score(y, tx, w, threshold=0):
    """calculates the final score of a given method"""
    labels = predict_regression_labels(w, tx, threshold)
    count = 0
    for i in range(len(labels)):
        if labels[i] == y[i]:
            count += 1
    accuracy = count/len(labels)
#     recall = count/
    return count/len(labels)

def logistic_score(y, tx, w, threshold=0.5):
    """Generates class predictions given weights, and a test data matrix
     for logistic regression methods"""
    labels = predict_logistic_labels(w, tx, threshold)
    count = 0
    for i in range(len(labels)):
        if labels[i] == y[i]:
            count += 1
    return count/len(labels)