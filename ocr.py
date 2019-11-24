
from chain_utils import *
from chain_featuremap import chain_featuremap
from solverFW import solverFW 
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
        'featureFn': chain_featuremap
    }

    model, progress = solverFW(param, options)

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

    plt.plot(progress['eff_pass'], progress['primal'], 'r-')
    plt.plot(progress['eff_pass'], progress['dual'], 'b--')
    plt.xlabel("Effective passes")
    plt.savefig("ocr.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCFW")

    parser.add_argument('--lambda', type=float, default=1e-2)
    parser.add_argument('--gap-threshold', type=float, default=0.1)
    parser.add_argument('--num-passes', type=int, default=100)
    parser.add_argument('--do-line-search', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--time-budget', type=int, default=100000000)

    options = parser.parse_args()
    ocr(options, 'data/letter.data')