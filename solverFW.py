
from helpers import *
import numpy as np
import time
from IPython import embed

eps = 1e-6

def get_default_option(n):
    return {
        'num_passes': 200,
        'do_line_search': True,
        'time_budget': np.inf,
        'debug': 0,
        'lambda': 1. / n,
        'test_data': [],
        'gap_threshold': 0.1
    }


def solverFW(param, options=None):
    phi = param['featureFn']
    loss = param['lossFn']
    maxOracle = param['oracleFn']

    patterns = param['patterns']
    labels = param['labels']
    n = len(patterns)

    if options is None:
        options = get_default_option(n)
    else:
        options = vars(options)
    lambd = options['lambda']
    d = len(phi(param, patterns[0], labels[0]))

    # Init
    model = {}
    progress = {}
    model['w'] = np.zeros((d, ))
    w_mat = np.zeros((d, n))

    model['l'] = 0.

    progress['primal'] = []
    progress['dual'] = []
    progress['eff_pass'] = []
    progress['train_error'] = []
    # TODO: test
    if 'test_data' in options and isinstance(options['test_data'], dict) and 'patterns' in options['test_data']:
        progress['test_error'] = []

    print("Running batch FW on {} examples. The options are as follows: {}".format(len(patterns), options))

    tic = time.time()
    # Main loop
    for k in range(options['num_passes']):
        w_s = np.zeros((d, ))
        l_s = 0
        for i in range(n):
            ystar_i = maxOracle(param, model, patterns[i], labels[i])
            psi_i = phi(param, patterns[i], labels[i]) - phi(param, patterns[i], ystar_i)
            # print(ystar_i)
            w_s += 1. / (lambd * n) * psi_i
            loss_i = loss(param, labels[i], ystar_i)
            l_s += 1. / n * loss_i
            assert((loss_i - model['w'].dot(psi_i) >= -1e-12)) # San check

        gap = lambd * (model['w'].dot(model['w'] - w_s)) - (model['l'] - l_s)
        if options['do_line_search']:
            gamma_opt = gap / (lambd * ((model['w'] - w_s).dot(model['w'] - w_s) + eps))
            gamma = max(0, min(1, gamma_opt))
        else:
            gamma = 2. / (k + 2)

        if gap <= options['gap_threshold']:
            print("Duality gap below threshold -- stopping!")
            print("Current gap: {}, gap_threshold: {}\n".format(gap, options["gap_threshold"]))
            print("Reached at iteration {}".format(k))
            break
        else:
            print("Duality gap check: gap = {} at iteration {}".format(gap, k))

        model['w'] = (1 - gamma) * model['w'] + gamma * w_s
        model['l'] = (1 - gamma) * model['l'] + gamma * l_s

        if options['debug']:
            f = -objective_function(model['w'], model['l'], lambd)
            gap, _, __ = duality_gap(param, maxOracle, model, lambd)
            primal = f + gap
            train_error = average_loss(param, maxOracle, model)

            print("Pass {} (iteration {}), SVM primal = {:.6f}, SVM dual = {:.6f}, duality gap = {:.6f}, train error = {:.6f}".format(
                k+1, k+1, primal, f, gap, train_error))

            progress['primal'].append(primal)
            progress['dual'].append(f)
            progress['eff_pass'].append(k)
            progress['train_error'].append(train_error)

            if 'test_data' in options and isinstance(options['test_data'], dict) and 'patterns' in options['test_data']:
                param_debug = param.copy()
                param_debug['patterns'] = options['test_data']['patterns']
                param_debug['labels'] = options['test_data']['labels']
                test_error = average_loss(param_debug, maxOracle, model)
                progress['test_error'].append(test_error)

        if (time.time() - tic) > options['time_budget']:
            print("Time budget exceeded.")
            return

    return model, progress