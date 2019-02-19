# -*- coding: utf-8 -*-
"""Implemented ML Methods.

least_squares_GD(*args)        \t Linear regression using gradient descent
least_squares_SGD(*args)       \t Linear regression using stochastic gradient descent
least_squares(*args)           \t Least squares regression using normal equations
ridge_regression(*args)        \t Ridge regression using normal equations
logistic_regression(*args)     \t Logistic regression using gradient descent
reg_logistic_regression(*args) \t Regularized logistic regression using gradient descent
"""
import numpy as np
from costs import compute_loss,compute_gradient
# from proj1_helpers import batch_iter
from gradient_optimization import update_gd, update_sgd



def least_squares(y, tx, loss_method='mse'):
    """calculate the least squares solution."""
    # weights
    w = np.linalg.solve(((tx.T).dot(tx)), (tx.T).dot(y))
    # loss, default using MSE
    loss = compute_loss(y, tx, w, loss_method=loss_method);
    return w,loss

def ridge_regression(y, tx, lambda_, loss_method='mse'):
    """implement ridge regression.
    Minimization of the penalized mean squared error with the ridge regularization.
    """
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    # weights
    w = np.linalg.solve(tx.T.dot(tx) + aI, tx.T.dot(y))
    # loss, default using MSE
    loss = compute_loss(y, tx, w, loss_method=loss_method);
    return w,loss

def least_squares_GD(y, tx, initial_w, max_iters, gamma, method='mse'):
    """Linear regression using gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    # Iterate to update w and loss
    print_head()
    for n_iter in range(max_iters):
        grad, w ,loss, ws, losses = update_gd(y, tx, w, ws, losses, gamma, method=method)
        print_step(n_iter, 10, max_iters, loss, w, method = 'gradient descent')
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1, method='mse'):
    """Linear regression using stochastic gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    # Iterate to update w and loss
    print_head()
    for n_iter in range(max_iters):
        grad, w ,loss, ws, losses = update_sgd(
            y, tx, w, ws, losses, gamma, batch_size=batch_size, method=method)
        print_step(n_iter, 10, max_iters, loss, w, method = 'stochastic gradient descent')
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # init parameters
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    # start the logistic regression
    print_head()
    for n_iter in range(max_iters):
        # using GD
        grad, w ,loss, ws, losses = update_gd(y, tx, w, ws, losses, gamma, method='sigmoid')
        # using SGD
#         grad, w ,loss, ws, losses = update_sgd(y, tx, w, ws, losses, gamma, 
#                                                batch_size=1, method='sigmoid')
        print_step(n_iter, 10, max_iters, loss, w, method = 'logistic regression')
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # init parameters
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    # start the logistic regression
    print_head()
    for n_iter in range(max_iters):
        # using GD
        grad, w ,loss, ws, losses = update_gd(y, tx, w, ws, losses,
                                              gamma, lambda_, method='sigmoid')
        # using SGD
        # grad, w ,loss, ws, losses = update_sgd(y, tx, w, ws, losses, gamma, lambda_,
        # batch_size=1, method='sigmoid')
        print_step(n_iter, 10, max_iters, loss, w, method = 'regularized logistic regression')
    return w, loss

def print_step(n_iter, step_wise, max_iters, loss, w, method = 'unknown'):
    if (n_iter%10 == 0) or n_iter == max_iters-1:
        print('{:<30s}{:>5d}/{:<5d}  {:^10.3e} {:^10.3e} {:^10.3e}'.format(
            method, n_iter, max_iters - 1, loss, w[0][0], w[1][0]))
def print_head():
    dash = '+' * 76
    print(dash)
    print('{:<30s} {:^11s}  {:^10s} {:^10s} {:^10s}'.format(
        'method', 'step', 'loss', 'w0', 'w1'))
    print(dash)