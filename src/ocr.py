
from chain_utils import *
from chain_featuremap import chain_featuremap
from solverBCFW import solverBCFW
from solverFW import solverFW 
from solverSSG import solverSSG
from loadOCR import loadOCRData

import argparse
import numpy as np 
import matplotlib.pyplot as plt

from IPython import embed

def ocr(options, path):
    patterns_train, labels_train, patterns_test, labels_test = loadOCRData(path)
    param = {
        'patterns': patterns_train,
        'labels': labels_train,
        'lossFn': chain_loss,
        'oracleFn': chain_oracle,
        'featureFn': chain_featuremap,
        'test_data':{
            'patterns': patterns_test,
            'labels': labels_test
        }
    }
    import os
    if not os.path.exists('../plot'):
        os.makedirs('../plot')
    if not os.path.exists('../output'):
        os.makedirs('../output')

    for exp_idx in range(options.repeat):

        print("Exp no {}".format(exp_idx))

        if options.method == 'ssg':
            model, progress = solverSSG(param, options)
        elif options.method == 'fw':
            model, progress = solverFW(param, options)
        elif options.method =='bcfw':
            model, progress = solverBCFW(param, options)
        else:
            print("Invalid method")
            return

        avg_loss = 0.
        n = len(patterns_train)
        for i in range(n):
            ypredict = chain_oracle(param, model, patterns_train[i])
            avg_loss += chain_loss(param, labels_train[i], ypredict)
        avg_loss /= n
        print("Average loss on the training set: {:.6f}".format(avg_loss))

        avg_loss = 0.
        n = len(patterns_test)
        for i in range(n):
            ypredict = chain_oracle(param, model, patterns_test[i])
            avg_loss += chain_loss(param, labels_test[i], ypredict)
        avg_loss /= n
        print("Average loss on the test set: {:.6f}".format(avg_loss))

        plt.clf()
        plt.plot(progress['eff_pass'], np.log10(progress['primal']), 'r-', label="primal")
        if 'dual' in progress:
            plt.plot(progress['eff_pass'], np.log10(progress['dual']), 'b--', label='dual')
            np.savez("../output/{}_{}.npz".format(options.name, exp_idx), primal=progress['primal'], dual=progress['dual'], eff_pass=progress['eff_pass'], test_error=progress['test_error'])
        else:
            np.savez("../output/{}_{}.npz".format(options.name, exp_idx), primal=progress['primal'], eff_pass=progress['eff_pass'], test_error=progress['test_error'])
        plt.xlabel("Effective passes")
        plt.ylabel("primal/dual in log10")
        plt.title("{}".format(options.name))
        plt.legend()
        plt.savefig("../plot/ocr_{}_train_exp{}.png".format(options.name, exp_idx))
        # plt.show()

        plt.clf()
        plt.plot(progress['eff_pass'], progress['test_error'], 'g--', label="test error")
        plt.xlabel("Effective passes")
        plt.ylabel("Test error")
        plt.title("{}".format(options.name))
        plt.legend()
        plt.savefig("../plot/ocr_{}_test_exp{}.png".format(options.name, exp_idx))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCFW")

    parser.add_argument('--lambda', type=float, default=1e-2)
    parser.add_argument('--gap-threshold', type=float, default=0.1)
    parser.add_argument('--num-passes', type=int, default=100)
    parser.add_argument('--do-line-search', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--time-budget', type=int, default=100000000)
    parser.add_argument('--do-weighted-averaging', action='store_true')
    parser.add_argument('--debug-multiplier', type=float, default=0)
    parser.add_argument('--sample', type=str, default='uniform')
    parser.add_argument('--gap-check', type=int, default=10)
    parser.add_argument('--method', choices=['ssg', 'fw', 'bcfw'])

    parser.add_argument('--name', type=str, default='tmp')
    parser.add_argument('--repeat', type=int, default=1)

    # Example:
    # python3 ocr.py --debug --method bcfw --do-line-search --do-weighted-averaging --name bcfw_wavg --repeat 10

    options = parser.parse_args()
    ocr(options, '../data/letter.data')