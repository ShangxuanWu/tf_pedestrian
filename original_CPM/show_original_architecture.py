import numpy as np

param_dict = np.load('./CPM-original.npy', encoding='latin1').item()

for key, value in param_dict.iteritems() :
    print key, value.shape