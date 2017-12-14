import numpy as np
''' 
    ReLU activation i.e. maximum(0, value)
'''
class ReLUActivation(object):
    #'forwarding'
    def apply(self, X):
        np.maximum(0, X)


    #gradient calculation
    def gradient(self, X):
        np.maximum(0, X)

'''
    Non activation i.e. value is invariant
'''
class NonActivation(object):
    # 'forwarding'
    def apply(self, X):
        return X

    # gradient calculation
    def gradient(self, X):
        return X
