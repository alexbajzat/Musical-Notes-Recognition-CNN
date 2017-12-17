import numpy as np


class SoftMax(object):
    def __init__(self, datasetSize, hyperParams):
        self.__datasetSize = datasetSize
        self.__hyperParams = hyperParams

    # calculates the prediction
    # returns the gradient on scores
    # todo add regularization
    def compute(self, X, labels):
        exponentiatedScores = np.exp(X)
        probabilites = exponentiatedScores / np.sum(exponentiatedScores, axis=1 ,keepdims=True)

        # we want to increase the loss of the bad predicted classes, so e log only the correct class
        # log(1) ~=1  => we stimulate the correct class with negative value
        correct = - np.log(probabilites[range(self.__datasetSize, labels)])
        dataLoss = np.sum(correct)/ self.__datasetSize
        print('loss: ' + dataLoss)

        #calculate the derivative
        derivativeProbs = probabilites
        derivativeProbs[range(self.__datasetSize), labels] -= 1
        derivativeProbs /= self.__datasetSize

        return derivativeProbs



