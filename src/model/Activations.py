import numpy as np

''' 
    ReLU activation i.e. maximum(0, value)
'''


class ReLUActivation(object):
    # 'forwarding'
    def apply(self, X):
        np.maximum(0, X)

    # derivative calculation
    # threshold the input values to 0 if smaller
    def derivative(self, X, gradients):
        gradients[X <= 0] = 0
        return gradients

'''
    Non activation i.e. value is invariant
'''


class NonActivation(object):
    # 'forwarding'
    def apply(self, X):
        return X

    # derivative calculation
    def derivative(self, X, gradients):
        return gradients

