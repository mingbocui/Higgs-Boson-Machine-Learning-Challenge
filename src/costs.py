# -*- coding: utf-8 -*-
"""a function used to compute the loss.
sigmoid(t)
calculate_sigmoid(y, tx, w)
calculate_mse(err)
calculate_mae(err)
compute_loss(y, tx, w, loss_method='mse')
compute_gradient(y, tx, w, loss_method='mse')
"""

import numpy as np

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_sigmoid(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    #loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    #return np.squeeze(- loss)
    return np.mean((-y * np.log(pred) - (1 - y) * np.log(1 - pred)))

def calculate_mse(err):
    """Calculate the mse for vector err."""
    return 1/2*np.mean(err**2)


def calculate_mae(err):
    """Calculate the mae for vector err."""
    return np.mean(np.abs(err))


def compute_loss(y, tx, w, loss_method='mse'):
    """Calculate the loss using mse, mae, rmse or sigmoid.
    """
    err = y - tx.dot(w)
    if loss_method == 'mse':
        return calculate_mse(err)
    elif loss_method == 'mae':
        return calculate_mae(err)
    elif loss_method == 'rmse':
        return np.sqrt(2 * calculate_mae(err))
    elif loss_method == 'sigmoid':
        return calculate_sigmoid(y, tx, w)
    else:
        raise NotImplementedError
        
def compute_gradient(y, tx, w, loss_method='mse'):
    """Compute the gradient of the cost function for linear regression
    you can use mse or mae
    """
    err = y - tx.dot(w)
    if loss_method == 'mse':
        grad = -tx.T.dot(err) / len(err)
    elif loss_method == 'mae':
        grad = -tx.T.dot(np.sign(err))/ len(err)
    elif loss_method == 'rmse':
        grad = -tx.T.dot(err/np.sqrt(2 * calculate_mse(err))) / len(err)
    elif loss_method == 'sigmoid':
        grad = -tx.T.dot(y - sigmoid(tx.dot(w)))
    else:
        raise NotImplementedError
    return grad