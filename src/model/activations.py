from copy import deepcopy
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
        applied = deepcopy(X)
        applied[applied <= 0] = 0
        return applied

    # derivative calculation
    # threshold the input values to 0 if smaller
    def derivative(self, X):
        derivatives = deepcopy(X)
        derivatives[derivatives <= 0] = 0
        derivatives[derivatives > 0] = 1
        return derivatives


'''
    Non activation i.e. value is invariant
'''


class NonActivation(Activation):
    def __init__(self):
        super().__init__(ActivationType.NONE)

    # 'forwarding'
    def apply(self, X):
        return deepcopy(X)

    # derivative calculation
    def derivative(self, X):
        return deepcopy(X)
