from copy import deepcopy

import numpy as np

from src.data.constants import LayerType
from src.utils.processing import exportPNGs, exportHistory, exportModel


class Model(object):

    def __init__(self, layers, classifier, batchSize, iterations=100):
        self.__batchSize = batchSize
        self.__classifier = classifier
        self.__layers = layers
        self.__iterations = iterations
        self.__history = []

    def train(self, dataset, validationSet):
        rawData = dataset[0]
        labels = dataset[1]
        start = 0
        endBatch = self.__batchSize
        while (True):
            batchedData = rawData[start:endBatch]
            batchedLabels = labels[start:endBatch]
            print('batch: ', str(start), ' - ', str(endBatch))

            for epoch in range(self.__iterations):
                print("epoch: ", epoch)
                data = batchedData
                weights = []

                # forward propagation
                for layer, type in self.__layers:
                    data = layer.forward(data)

                    if (type == LayerType.HIDDEN or type == LayerType.CONV):
                        weights.append(layer.getWeights())

                    if (epoch > (90 / 100 * self.__iterations) and type == LayerType.CONV):
                        exportPNGs(data[0], str(type) + " " + str(batchedLabels[0]))
                # scores
                back, loss = self.__classifier.compute(data, batchedLabels, weights)

                # backpropagation
                for layer, type in reversed(self.__layers):
                    back = layer.backprop(back)

                # validate
                acc = self.validate(validationSet)

                # keep for later export
                self.__history.append((loss, acc))
                print('\n')

            # next batch
            start += self.__batchSize
            endBatch = start + self.__batchSize
            if (endBatch >= len(rawData)):
                break

        # create html report
        self.__saveHistory()
        sample = self.predict(rawData[0])
        self.__saveModel(sample)

    def __saveFeatures(self, convLayer):
        filters, number, size, depth = convLayer.getFilters()
        parsed = filters.reshape(number * depth, size, size)
        exportPNGs(parsed, 'filter-conv1')

    def __saveHistory(self):
        exportHistory((self.__history, 'conf'))

    def __saveModel(self, sample):
        exportModel(self.__layers, sample)

    def validate(self, dataset):
        data = dataset[0]
        labels = dataset[1]

        # forward propagation
        for layer, type in self.__layers:
            data = layer.forward(data)

        # scores
        predictedClasses = np.argmax(data, axis=1)
        mean = np.mean(predictedClasses == np.transpose(labels))
        print('training accuracy:', (mean))
        return mean

    '''
        predict and return 3-uple (data, raw result, probabilities)
    '''
    def predict(self, data):
        pre = deepcopy(data)

        if data.shape != 3:
            data = np.expand_dims(data, 1)

        # forward propagation
        for layer, type in self.__layers:
            data = layer.forward(data)

        # predict
        exponentiatedScores = np.exp(data)

        # normalize probabilities
        probabilites = exponentiatedScores / np.sum(exponentiatedScores, axis=1, keepdims=True)

        # remove additional useless dim
        return pre, data[0], probabilites[0]
