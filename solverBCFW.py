from helpers import *
import numpy as np
import time
from IPython import embed

eps = 1e-6

def get_default_option(n):
    return {
        'num_passes': 200,
        'do_line_search': True,
        'do_weighted_averaging': True,
        'time_budget': np.inf,
        'debug': False,
        'rand_seed': 1,
        'sample': 'uniform',
        'debug_multiplier': 0,
        'lambda': 1. / n,
        'test_data': [],
        'gap_threshold': 0.1,
        'gap_check': 10
    }

def solverBCFW(param, options=None):

    phi = param['featureFn']
    loss = param['lossFn']
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

    l = 0
    l_mat = np.zeros((n, ))

    if options['do_weighted_averaging']:
        w_avg = model['w']
        l_avg = 0.
    if options['debug_multiplier'] == 0:
        debug_iter = n
        options['debug_multiplier'] = 100
    else:
        debug_iter = 1

    progress['primal'] = []
    progress['dual'] = []
    progress['eff_pass'] = []
    progress['train_error'] = []
    if 'test_data' in options and isinstance(options['test_data'], dict) and 'patterns' in options['test_data']:
        progress.test_error = []

    print("Running BCFW on {} examples. The options are as follows:\n{}".format(len(patterns), options))

    tic = time.time()
    k = 0
    gap_check_counter = 1
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
            loss_i = loss(param, labels[i], ystar_i)
            l_s = 1. / n * loss_i
            assert ((loss_i - model['w'].dot(psi_i)) >= -1e-12)

            if options['do_line_search']:
                gamma_opt = (model['w'].dot(w_mat[:, i] - w_s) - 1. / lambd * (l_mat[i] - l_s)) / ((w_mat[:, i] - w_s).dot(w_mat[:, i] - w_s) + eps)
                gamma = max(0, min(1, gamma_opt))
            else:
                gamma = 2. * n / (k + 2 * n)

            model['w'] = model['w'] - w_mat[:, i]
            w_mat[:, i] = (1 - gamma) * w_mat[:, i] + gamma * w_s
            model['w'] = model['w'] + w_mat[:, i]

            l = l - l_mat[i]
            l_mat[i] = (1 - gamma) * l_mat[i] + gamma * l_s
            l = l + l_mat[i]

            if options['do_weighted_averaging']:
                rho = 2. / (k + 2)
                w_avg = (1 - rho) * w_avg + rho * model['w']
                l_avg = (1 - rho) * l_avg + rho * l

            k += 1
            if options['debug'] and k >= debug_iter:
                model_debug = {}
                model_debug['w'] = w_avg if options['do_weighted_averaging'] else model['w']
                model_debug['l'] = l_avg if options['do_weighted_averaging'] else l

                f = -objective_function(model_debug['w'], model_debug['l'], lambd)
                gap = duality_gap(param, maxOracle, model_debug, lambd)
                primal = f + gap
                train_error = average_loss(param, maxOracle, model_debug)
                print("Pass {} (iteration {}), SVM primal = {:.6f}, SVM dual = {:.6f}, duality gap = {:.6f}, train error = {:.6f}".format(
                    k+1, k+1, primal, f, gap, train_error))

                progress['primal'].append(primal)
                progress['dual'].append(dual)
                progress['eff_pass'].append(k * 1. / n)
                progress['train_error'].append(train_error)
                if 'test_data' in options and isinstance(options['test_data'], dict) and 'patterns' in options['test_data']:
                    param_debug = param.copy()
                    param_debug['patterns'] = options['test_data'].patterns
                    param_debug['labels'] = options['test_data'].labels
                    test_error = average_loss(param_debug, maxOracle, model_debug)
                    progress['test_error'].append(test_error)

                debug_iter = min(debug_iter + n, int(debug_iter + (1 + options['debug_multiplier'] / 100.)))
            t_elapsed = time.time() - tic
            if (t_elapsed / 60. > options['time_budget']):
                print("Time budget exceeded.")
                if options['do_weight_averaging']:
                    model['w'] = w_avg
                    model['l'] = l_avg
                else:
                    model['l'] = l
                return model, progress

        # Outside the dummy loop
        if options['gap_check'] and gap_check_counter >= options['gap_check']:
            gap_check_counter = 0
            model_debug = {}
            model_debug['w'] = w_avg if options['do_weighted_averaging'] else model['w']
            model_debug['l'] = l_avg if options['do_weighted_averaging'] else l

            gap = duality_gap(param, maxOracle, model_debug, lambd)
            if gap <= options['gap_threshold']:
                print("Duality gap below threshold -- stopping!")
                print("Current gap: {}, gap_threshold: {}\n".format(gap, options["gap_threshold"]))
                print("Reached after iteration {}".format(k))
                break
            else:
                print("Duality gap check: gap = {} at iteration {}".format(gap, k))
        gap_check_counter += 1

    if options['do_weighted_averaging']:
        model['w'] = w_avg
        model['l'] = l_avg
    else:
        model['l'] = l

    return model, progress