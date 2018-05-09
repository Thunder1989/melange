'''
to test each of the data feature functions and output the dimension of each feature set
'''
from data_feature_extractor import *

import numpy as np


X = np.random.rand(10,100)
data_feature_tester = data_feature_extractor(X)

res = map(lambda x: eval('data_feature_tester.' + x + '()'), data_feature_tester.functions)

for F, fun in zip(res, data_feature_tester.functions):
    print fun, 'gives', F.shape

