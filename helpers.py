import numpy as np
from IPython import embed

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
        ystars.append(maxOracle(param, model, patterns[i], labels[i], debug=False))

    w_s = np.zeros((len(w), ))
    l_s = 0.
    for i in range(n):
        w_s += phi(param, patterns[i], labels[i]) - phi(patterns, patterns[i], ystars[i])
        l_s += loss(param, labels[i], ystars[i])
    
    w_s /= lambd * n
    l_s /= n
    gap = lambd * w.dot(w - w_s) - l + l_s
    # embed()
    return gap, w_s, l_s
