import numpy as np


class SoftMax(object):
    def __init__(self, datasetSize, hyperParams):
        self.__datasetSize = datasetSize
        self.__hyperParams = hyperParams

    # calculates the prediction
    # returns the gradient on scores
    # todo add regularization
    def compute(self, X, labels):
        # we want to calculate loss using cross-entropy
        # calculate the probabilities of class
        exponentiatedScores = np.exp(X)
        # normalize probabilities
        probabilites = exponentiatedScores / np.sum(exponentiatedScores, axis=1, keepdims=True)
        # cross-entropy
        correct = - np.log(probabilites[range(self.__datasetSize), np.asarray(labels)])
        dataLoss = np.sum(correct) / self.__datasetSize
        print('loss: ', dataLoss)

        # calculate the derivative
        derivativeProbs = probabilites
        derivativeProbs[range(self.__datasetSize), np.array(labels)] -= 1
        derivativeProbs /= self.__datasetSize

        return derivativeProbs
