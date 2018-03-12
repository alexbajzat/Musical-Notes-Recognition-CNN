from enum import Enum

''' 
    ReLU activation i.e. maximum(0, value)
'''


class ActivationType(Enum):
    RELU = "ReLU",
    NONE = "None"


class Activation(object):
    def __init__(self, type):
        self.__type = type

    def getType(self):
        return self.__type


class ReLUActivation(Activation):
    def __init__(self):
        super().__init__(ActivationType.RELU)

    # 'forwarding'
    def apply(self, X):
        X[X <= 0] = 0
        return X

    # derivative calculation
    # threshold the input values to 0 if smaller
    def derivative(self, X, gradients):
        gradients[X <= 0] = 0
        return gradients


'''
    Non activation i.e. value is invariant
'''


class NonActivation(Activation):
    def __init__(self):
        super().__init__(ActivationType.NONE)

    # 'forwarding'
    def apply(self, X):
        return X

    # derivative calculation
    def derivative(self, X, gradients):
        return gradients
