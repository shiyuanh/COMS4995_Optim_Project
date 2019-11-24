import numpy as np
from loadOCR import convert_ocr, load_ocr, loadOCRData

def chain_featuremap(param, x, y):
    data = np.array(x['data'])
    #num_vars, num_dims = data.shape
    num_dims, num_vars = data.shape
    num_states = x['num_states']

    phi = np.zeros(num_states*num_dims+2*num_states+num_states**2)
    for i in range(num_vars):
        idx = y[i] * num_dims
        #phi[idx : idx+num_dims] = phi[idx : idx+num_dims] + data[i, :]
        phi[idx : idx+num_dims] = phi[idx : idx+num_dims] + data[:, i]
    phi[num_states*num_dims+y[0]] = 1
    phi[num_states*num_dims+num_states+y[-1]] = 1

    offset = num_states*num_dims + 2*num_states
    for i in range(num_vars - 1):
        idx = y[i] + num_states * y[i+1]
        phi[offset + idx] = phi[offset + idx] + 1

    #print(phi)
    #print(sum(phi))
    #print(np.nonzero(phi))
    #print(y)
    return phi


if __name__ == '__main__':
    param = None
    ocr = 'data/letter.data'

    patterns_train, labels_train, patterns_test, labels_test = loadOCRData(ocr)
    x = patterns_train[0]
    y = labels_train[0]

    chain_featuremap(param, x, y)
