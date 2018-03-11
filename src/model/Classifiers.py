import numpy as np


class SoftMax(object):
    def __init__(self, hyperParams):
        self.__hyperParams = hyperParams

    # calculates the prediction
    # returns the gradient on scores
    def compute(self, X, labels, weights):
        inputNumber = X.shape[0]
        # we want to calculate loss using cross-entropy
        # calculate the probabilities of class
        exponentiatedScores = np.exp(X)

        # normalize probabilities
        probabilites = exponentiatedScores / np.sum(exponentiatedScores, axis=1, keepdims=True)

        # cross-entropy
        correct = -np.log(probabilites[range(inputNumber), np.transpose(labels)])
        dataLoss = np.sum(correct) / inputNumber
        regularizationLoss = 0
        for weight in weights:
            regularizationLoss += 0.5 * self.__hyperParams.regularization * np.sum(weight * weight)

        # total loss data loss + regularization loss
        loss = dataLoss + regularizationLoss
        print('loss: ', loss)

        # calculate the derivative
        derivativeProbs = probabilites
        derivativeProbs[range(inputNumber), np.transpose(labels)] -= 1
        derivativeProbs /= inputNumber

        return derivativeProbs, loss
