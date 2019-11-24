# -*- coding: utf-8 -*-
# @Author: yuchen
# @Date:   2019-11-23 16:50:13
# @Last Modified by:   yuchen
# @Last Modified time: 2019-11-23 19:25:51

import numpy as np

def objective_function(w, b_alpha, lambd):
    return lambd / 2. * w.dot(w) - b_alpha

def average_loss(param, maxOracle, model):
    patterns = param['patterns']
    labels = param['labels']
    loss = param['lossFn']

    loss_term = 0.
    n = len(patterns)
    for i in range(n):
        ystar_i = maxOracle(param, model, patterns[i])
        loss_term += loss(param, labels[i], ystar_i)
    loss_term /= n
    return loss_term

def duality_gap(param, maxOracle, model, lambd):
    patterns = param['patterns']
    labels = param['labels']
    loss = param['lossFn']
    phi = param['featureFn']
    
    w = model['w']
    l = model['l']
    
    n = len(patterns)
    ystars = []
    for i in range(n):
        ystars.append(maxOracle(param, model, patterns[i], labels[i]))

    w_s = np.zeros((w, ))
    l_s = 0.
    for i in range(n):
        w_s += (phi(param, patterns[i], labels[i]) - phi(patterns, patterns[i], ystars[i]))
        l_s += loss(param, labels[i], ystars[i])
    
    w_s /= lambd * n
    l_s /= n
    gap = lambd * w.dot(w - w_s) - l + l_s
    
    return gap, w_s, l_s
