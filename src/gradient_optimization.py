# -*- coding: utf-8 -*-
"""Implementing GD/SGD updating
"""
import numpy as np
from costs import compute_loss,compute_gradient
from proj1_helpers import batch_iter

def update_gd(y, tx, w, ws, losses, gamma, lambda_=0, method='mse'):
    # compute gradient
    grad = compute_gradient(y, tx, w, loss_method=method) + lambda_ * w
    # gradient w by descent update
    w = w - gamma * grad
    # calculate loss
    loss = compute_loss(y, tx, w, loss_method=method) + 0.5* lambda_ * np.sum(w**2)
    ws.append(w)
    losses.append(loss)
    return grad, w ,loss, ws, losses

def update_sgd(y, tx, w, ws, losses, gamma, lambda_=0, batch_size=1, method='mse'):
    for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
        # compute a stochastic gradient
        grad = compute_gradient(y_batch, tx_batch, w, loss_method=method) + lambda_ * w
        # update w through the stochastic gradient update
        w = w - gamma * grad
        # calculate loss
        loss = compute_loss(y, tx, w, loss_method=method) + 0.5* lambda_ * np.sum(w**2)
        # store w and loss
        ws.append(w)
        losses.append(loss)
    return grad, w ,loss, ws, losses