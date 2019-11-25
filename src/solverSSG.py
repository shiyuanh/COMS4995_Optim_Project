from helpers import *
import numpy as np
import time

eps = 1e-6

def get_default_option(n):
    return {
        'num_passes': 50,
        'do_weighted_averaging': True,
        'do_line_search': True,
        'time_budget': np.inf,
        'debug': False,
        'rand_seed': 1,
        'sample': 'uniform',
        'debug_multiplier': 0,
        'lambda': 1. / n,
        # 'test_data': [],
    }

def solverSSG(param, options=None):

    phi = param['featureFn']
    maxOracle = param['oracleFn']

    patterns = param['patterns']
    labels = param['labels']
    n = len(patterns)

    options = get_default_option(n) if options is None else vars(options)
    lambd = options['lambda']
    phi1 = phi(param, patterns[0], labels[0])
    d = len(phi1)
    model = {}
    progress = {}

    model['w'] = np.zeros((d, ))
    w_mat = np.zeros((d, n))

    if options['do_weighted_averaging']:
        w_avg = model['w']
    if options['debug_multiplier'] == 0:
        debug_iter = n
        options['debug_multiplier'] = 1
    else:
        debug_iter = 1

    progress['primal'] = []
    progress['eff_pass'] = []
    progress['train_error'] = []
    if 'test_data' in param and isinstance(param['test_data'], dict) and 'patterns' in param['test_data']:
        progress['test_error'] = []

    print("Running SSG on {} examples. The options are as follows:\n{}".format(len(patterns), options))
    tic = time.time()

    k = 0
    for p in range(options['num_passes']):
        perm = np.random.permutation(n)
        for dummy in range(n):
            if options['sample'] == 'uniform':
                i = np.random.randint(n)
            elif options['sample'] == 'perm':
                i = perm[dummy]
            else:
                print("Illegal sample option")
                return

            ystar_i = maxOracle(param, model, patterns[i], labels[i])
            psi_i = phi(param, patterns[i], labels[i]) - phi(param, patterns[i], ystar_i)
            w_s = 1. / (n * lambd) * psi_i

            gamma = 1. / (k + 1)
            model['w'] = (1 - gamma) * model['w'] + gamma * n * w_s

            if options['do_weighted_averaging']:
                rho = 2. / (k + 2)
                w_avg = (1 - rho) * w_avg + rho * model['w']
            k += 1
            if options['debug'] and k == debug_iter:
                model_debug = {}
                model_debug['w'] = w_avg if options['do_weighted_averaging'] else model['w'].copy()

                primal = primal_objective(param, maxOracle, model_debug, lambd)
                train_error = average_loss(param, maxOracle, model_debug)
                print("Pass {} (iteration {}), SVM primal = {:.6f}, train error = {:.6f}".format(
                    int(k/n), k+1, primal, train_error))

                progress['primal'].append(primal)
                progress['eff_pass'].append(k * 1. / n)
                progress['train_error'].append(train_error)
                if 'test_data' in param and isinstance(param['test_data'], dict) and 'patterns' in param['test_data']:
                    param_debug = param.copy()
                    param_debug['patterns'] = param['test_data']['patterns']
                    param_debug['labels'] = param['test_data']['labels']
                    test_error = average_loss(param_debug, maxOracle, model_debug)
                    progress['test_error'].append(test_error)

                debug_iter += min(1.0, options['debug_multiplier']) * n

            t_elapsed = time.time() - tic
            if (t_elapsed / 60. > options['time_budget']):
                print("Time budget exceeded.")
                if options['do_weight_averaging']:
                    model['w'] = w_avg
                return model, progress

    if options['do_weighted_averaging']:
        model['w'] = w_avg

    return model, progress
