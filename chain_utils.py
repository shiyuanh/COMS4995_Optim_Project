# -*- coding: utf-8 -*-
# @Author: yuchen
# @Date:   2019-11-23 18:48:11
# @Last Modified by:   yuchen
# @Last Modified time: 2019-11-23 19:19:10

import numpy as np 

def chain_loss(param, ytruth, ypredict):
	return sum(ypredict != ytruth) / len(ytruth)

def chain_oracle(param, model, xi, yi=None):
	w = model['w']
	num_dims = xi['data'].shape[0]
	num_vars = xi['data'].shape[1]
	num_states = xi['num_states']

	weight = weightVec2Cell(w, num_states, num_dims)
	theta_unary = np.matmul(weight[0].T, xi['data'])

	theta_unary[:, 0] = theta_unary[:, 0] + weight[1]
	theta_unary[:, -1] = theta_unary[:, -1] + weight[2]
	theta_pair = weight[3]

	if yi is not None:
		L = len(yi)
		for i in range(num_vars):
			theta_unary[:, i] = theta_unary[:, i] + 1. / L
			idx = yi[i]
			theta_unary[idx, i] = theta_unary[idx, i] - 1. / L

	label = chain_logDecode(theta_unary.T, theta_pair)
	label = label.T - 1
	return label

def weightVec2Cell(w, num_states, d):
	idx = num_states * d
	weight = []
	weight.append(np.reshape(w[:idx], (d, num_states)))
	weight.append(w[idx: idx + num_states])
	idx += num_states
	weight.append(w[idx: idx + num_states])
	idx += num_states
	weight.append(np.reshape(w[idx:], (num_states, num_states)))
	return weight

def chain_logDecode(logNodePot, logEdgePot):
	nNodes, nStates = logNodePot.shape[0], logNodePot.shape[1]
	alpha = np.zeros((nNodes, nStates))
	alpha[0, :] = logNodePot(0, :)
	mxState = [None]
	for n in range(1, nNodes):
		tmp = np.tile(alpha[n-1, :].T, (1, nStates)) + logEdgePot	# ???
		alpha[n, :] = logNodePot[n, :] + max(tmp)
		mxState.append(np.argmax(tmp))
	y = np.zeros((nNodes, 1))
	y[nNodes - 1] = np.argmax(alpha[nNodes - 1, :])
	for n in range(nNodes - 2, 1, -1):
		y[n] = mxState[n + 1, y[n + 1]]
	return y
