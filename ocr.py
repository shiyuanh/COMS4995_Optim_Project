
from chain_utils import *
from solverFW import solverFW 

import numpy as np 
import matplotlib.pyplot as plt

def ocr(patterns_train, labels_train, patterns_test, labels_test):

	param = {
		'patterns': patterns_train,
		'labels': labels_train,
		'lossFn': chain_loss,
		'oracleFn': chain_oracle,
		'featureFn': chain_featuremap
	}
	options = {
		'lambda': 1e-2,
		'gap_threshold': 0.1,
		'num_passes': 100,
		'do_line_search': True,
		'debug': True
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

